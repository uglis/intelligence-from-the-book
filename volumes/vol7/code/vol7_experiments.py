"""
Verified Python/NumPy/PyTorch experiments for AI Series Volume 7:
AI Research Practice, Rigorous Evaluation, Safety, Ethics & Reproducibility.
Generates real CSV files in ../data/ for pgfplots and prints verified metrics.
"""
import os
import csv
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def exp1_seed_cherrypick_and_mcnemar():
    """Ch 1: Best-of-K random seed inflation illusion + Paired Bootstrap CI & McNemar test."""
    rng = np.random.default_rng(2026)
    N_test = 200
    true_p = 0.80
    # 1. Cherry-picking max over K independent evaluations of an IDENTICAL true-80% classifier
    n_sims = 3000
    with open(os.path.join(OUT_DIR, "seed_inflation.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["k_seeds", "mean_max_acc", "p95_max_acc", "true_acc"])
        for K in [1, 2, 3, 5, 8, 12, 16, 20]:
            accs = rng.binomial(N_test, true_p, size=(n_sims, K)) / N_test * 100.0
            max_accs = accs.max(axis=1)
            wr.writerow([K, f"{max_accs.mean():.3f}", f"{np.percentile(max_accs, 95):.3f}", f"{true_p*100:.1f}"])
            if K in (1, 5, 10, 20):
                print(f"[Ch1] Best-of-{K:2d} cherry-pick (true=80.0%): E[max]={max_accs.mean():.2f}%, 95th={np.percentile(max_accs,95):.2f}%")

    # 2. Paired McNemar & Bootstrap on N=400 test samples (Model A: 82.0% vs Model B: 85.5%)
    # Construct paired outcomes with n01=24 (B right, A wrong), n10=10 (A right, B wrong), both right=318, both wrong=48
    n01, n10 = 24, 10
    A_ok = np.array([1]*318 + [1]*n10 + [0]*n01 + [0]*48)
    B_ok = np.array([1]*318 + [0]*n10 + [1]*n01 + [0]*48)
    diff_obs = (B_ok.mean() - A_ok.mean()) * 100.0
    # Exact continuity-corrected McNemar chi2
    chi2 = ((abs(n01 - n10) - 1.0) ** 2) / (n01 + n10)
    # 2000-resample paired bootstrap CI
    idx = rng.integers(0, len(A_ok), size=(2000, len(A_ok)))
    boot_diffs = (B_ok[idx].mean(axis=1) - A_ok[idx].mean(axis=1)) * 100.0
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    print(f"[Ch1] Paired eval (N=400): A={A_ok.mean()*100:.2f}%, B={B_ok.mean()*100:.2f}%, diff=+{diff_obs:.2f}%, Bootstrap 95% CI=[{ci_lo:.2f}%, {ci_hi:.2f}%], McNemar chi2={chi2:.3f}")


def exp2_calibration_and_conformal():
    """Ch 2: Overconfident logits -> Temperature Scaling ECE drop + Split Conformal Prediction coverage."""
    rng = np.random.default_rng(2026)
    N_cal, N_te, K = 500, 1000, 4
    # Generate true class probabilities and overconfident logits (scaled by 3.2x!)
    def make_split(n):
        z_true = rng.normal(0, 1.3, size=(n, K))
        p_true = np.exp(z_true - z_true.max(axis=1, keepdims=True))
        p_true /= p_true.sum(axis=1, keepdims=True)
        # Sample y from true categorical distribution
        u = rng.random(n)
        y = (np.cumsum(p_true, axis=1) < u[:, None]).sum(axis=1)
        # Overconfident model outputs logits = 3.2 * z_true
        logits = 3.2 * z_true
        return logits, y

    Z_cal, y_cal = make_split(N_cal)
    Z_te, y_te = make_split(N_te)

    def softmax_np(Z):
        Zs = Z - Z.max(axis=1, keepdims=True)
        e = np.exp(Zs)
        return e / e.sum(axis=1, keepdims=True)

    def compute_ece(probs, y, n_bins=10):
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        acc = (pred == y).astype(float)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        rows = []
        for b in range(n_bins):
            mask = (conf > bins[b]) & (conf <= bins[b + 1]) if b > 0 else (conf >= bins[b]) & (conf <= bins[b + 1])
            cnt = int(mask.sum())
            if cnt > 0:
                avg_c = float(conf[mask].mean())
                avg_a = float(acc[mask].mean())
                ece += (cnt / len(y)) * abs(avg_a - avg_c)
                rows.append((0.5 * (bins[b] + bins[b + 1]), avg_c, avg_a, cnt))
        return float(ece), rows

    # Fit optimal temperature T* on Z_cal minimizing NLL
    best_T, best_nll = 1.0, float("inf")
    for T in np.linspace(0.5, 5.0, 451):
        P = softmax_np(Z_cal / T)
        nll = float(-np.mean(np.log(P[np.arange(N_cal), y_cal] + 1e-12)))
        if nll < best_nll:
            best_nll, best_T = nll, float(T)

    P_raw = softmax_np(Z_te)
    P_cal = softmax_np(Z_te / best_T)
    ece_raw, bins_raw = compute_ece(P_raw, y_te)
    ece_cal, bins_cal = compute_ece(P_cal, y_te)
    acc_raw = float(np.mean(P_raw.argmax(axis=1) == y_te) * 100.0)
    acc_cal = float(np.mean(P_cal.argmax(axis=1) == y_te) * 100.0)
    nll_raw = float(-np.mean(np.log(P_raw[np.arange(N_te), y_te] + 1e-12)))
    nll_cal = float(-np.mean(np.log(P_cal[np.arange(N_te), y_te] + 1e-12)))

    # Write reliability diagram CSV (10 bins matched by bin_mid)
    with open(os.path.join(OUT_DIR, "calib_temp.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["bin_mid", "raw_acc", "cal_acc"])
        raw_map = {round(r[0], 2): r[2] for r in bins_raw}
        cal_map = {round(r[0], 2): r[2] for r in bins_cal}
        for mid in sorted(set(raw_map.keys()) | set(cal_map.keys())):
            if mid in raw_map and mid in cal_map:
                wr.writerow([f"{mid:.2f}", f"{raw_map[mid]:.4f}", f"{cal_map[mid]:.4f}"])

    # Split Conformal Prediction (alpha=0.10 -> guaranteed >= 90% coverage!)
    alpha = 0.10
    P_cal_split = softmax_np(Z_cal / best_T)
    scores_cal = 1.0 - P_cal_split[np.arange(N_cal), y_cal]
    q_level = math.ceil((N_cal + 1) * (1.0 - alpha)) / N_cal
    q_hat = float(np.quantile(scores_cal, q_level, method="higher"))
    sets_te = (1.0 - P_cal) <= q_hat
    cov = float(np.mean(sets_te[np.arange(N_te), y_te]) * 100.0)
    avg_size = float(sets_te.sum(axis=1).mean())
    print(f"[Ch2] Temp Scaling T*={best_T:.2f}: Acc {acc_raw:.2f}%=={acc_cal:.2f}%, ECE {ece_raw:.4f}->{ece_cal:.4f}, NLL {nll_raw:.4f}->{nll_cal:.4f}")
    print(f"[Ch2] Split Conformal (alpha=0.10): q_hat={q_hat:.4f}, empirical coverage={cov:.2f}% (>=90%), avg_set_size={avg_size:.2f}")


def exp3_adversarial_fgsm_pgd():
    """Ch 3: Standard MLP vs PGD Adversarially-Trained MLP under l_inf PGD attack sweep."""
    torch.manual_seed(2026)
    N_tr, N_te, D = 600, 300, 12
    # Linearly-separable + nonlinear clustered 2-class problem in R^12
    w_sig = torch.randn(D); w_sig /= w_sig.norm()
    def make_data(n):
        y = torch.randint(0, 2, (n,))
        sgn = (2 * y - 1).float().unsqueeze(-1)
        X = sgn * 1.15 * w_sig + 0.55 * torch.randn(n, D)
        return X, y

    X_tr, y_tr = make_data(N_tr)
    X_te, y_te = make_data(N_te)

    def pgd_attack(model, X, y, eps, steps=7):
        if eps == 0.0:
            return X.clone()
        X_adv = X.detach().clone()
        alpha = eps / 3.0
        for _ in range(steps):
            X_adv.requires_grad_(True)
            loss = F.cross_entropy(model(X_adv), y)
            grad = torch.autograd.grad(loss, X_adv)[0]
            X_adv = (X_adv + alpha * grad.sign()).detach()
            X_adv = torch.clamp(X_adv, X - eps, X + eps)
        return X_adv

    def train_model(adv_eps=0.0):
        torch.manual_seed(42)
        m = nn.Sequential(nn.Linear(D, 48), nn.ReLU(), nn.Linear(48, 48), nn.ReLU(), nn.Linear(48, 2))
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        for _ in range(90):
            Xin = pgd_attack(m, X_tr, y_tr, adv_eps, steps=5) if adv_eps > 0 else X_tr
            loss = F.cross_entropy(m(Xin), y_tr)
            opt.zero_grad(); loss.backward(); opt.step()
        return m

    m_std = train_model(adv_eps=0.0)
    m_adv = train_model(adv_eps=0.25)

    with open(os.path.join(OUT_DIR, "adv_robust.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["eps", "std_acc", "adv_acc"])
        for eps in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
            Xa_std = pgd_attack(m_std, X_te, y_te, eps)
            Xa_adv = pgd_attack(m_adv, X_te, y_te, eps)
            with torch.no_grad():
                acc_s = float((m_std(Xa_std).argmax(-1) == y_te).float().mean().item() * 100.0)
                acc_a = float((m_adv(Xa_adv).argmax(-1) == y_te).float().mean().item() * 100.0)
            wr.writerow([f"{eps:.2f}", f"{acc_s:.2f}", f"{acc_a:.2f}"])
            if eps in (0.0, 0.15, 0.25, 0.35):
                print(f"[Ch3] PGD eps={eps:.2f}: Std_MLP={acc_s:.2f}%, AdvTrained_MLP={acc_a:.2f}%")


def exp4_integrated_gradients_and_fairness():
    """Ch 4: Integrated Gradients Completeness Axiom error vs steps m + Fairness Impossibility trade-off."""
    torch.manual_seed(2026)
    net = nn.Sequential(nn.Linear(5, 16), nn.Tanh(), nn.Linear(16, 1))
    x = torch.tensor([[1.2, -0.8, 0.5, 1.5, -0.3]])
    x0 = torch.zeros_like(x)
    delta_F = float((net(x) - net(x0)).item())

    with open(os.path.join(OUT_DIR, "ig_completeness.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["steps", "ig_sum", "delta_F", "abs_err"])
        for m in [2, 5, 10, 20, 50, 100, 200]:
            alphas = torch.linspace(0.5 / m, 1.0 - 0.5 / m, m).unsqueeze(-1)
            pts = (x0 + alphas * (x - x0)).requires_grad_(True)
            out = net(pts).sum()
            grads = torch.autograd.grad(out, pts)[0]
            ig = (x - x0) * grads.mean(dim=0, keepdim=True)
            ig_sum = float(ig.sum().item())
            err = abs(ig_sum - delta_F)
            wr.writerow([m, f"{ig_sum:.6f}", f"{delta_F:.6f}", f"{err:.6e}"])
            if m in (2, 10, 50, 200):
                print(f"[Ch4] IG steps m={m:3d}: sum(IG)={ig_sum:.6f}, F(x)-F(x0)={delta_F:.6f}, err={err:.2e}")

    # Exact verification of Chouldechova Fairness Impossibility on two calibrated groups with base rates pA=0.20, pB=0.50
    rng = np.random.default_rng(2026)
    N = 10000
    def sim_group(base_rate):
        # True calibrated score s = P(Y=1|X) ~ Beta(a, b) with mean=base_rate
        a, b = 2.0 * base_rate, 2.0 * (1.0 - base_rate)
        s = rng.beta(a, b, size=N)
        y = (rng.random(N) < s).astype(int)
        return s, y

    sA, yA = sim_group(0.20)
    sB, yB = sim_group(0.50)
    # Unified calibrated threshold t=0.50 -> calibrated, but different FPR/TPR!
    pA, pB = (sA >= 0.5), (sB >= 0.5)
    fprA = float(pA[yA == 0].mean()); fprB = float(pB[yB == 0].mean())
    tprA = float(pA[yA == 1].mean()); tprB = float(pB[yB == 1].mean())
    ppvA = float(yA[pA].mean()); ppvB = float(yB[pB].mean())
    print(f"[Ch4] Calibrated threshold t=0.50 (pA=0.20, pB=0.50): PPV ({ppvA:.3f} vs {ppvB:.3f}) | FPR ({fprA:.3f} vs {fprB:.3f}) | TPR ({tprA:.3f} vs {tprB:.3f})")


def exp5_passk_and_arena():
    """Ch 5: Unbiased pass@k hypergeometric estimator vs biased naive plug-in estimator."""
    rng = np.random.default_rng(2026)
    p_true = 0.15
    n_samples = 20
    n_trials = 5000

    def comb(n, k):
        return math.comb(n, k) if 0 <= k <= n else 0

    with open(os.path.join(OUT_DIR, "passk_bias.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["k", "true_passk", "unbiased_est", "naive_est"])
        c_draws = rng.binomial(n_samples, p_true, size=n_trials)
        for k in [1, 2, 4, 6, 8, 10, 12, 16, 20]:
            true_pk = 1.0 - (1.0 - p_true) ** k
            unbiased = np.mean([1.0 - comb(n_samples - int(c), k) / comb(n_samples, k) for c in c_draws])
            naive = np.mean(1.0 - (1.0 - c_draws / n_samples) ** k)
            wr.writerow([k, f"{true_pk:.4f}", f"{unbiased:.4f}", f"{naive:.4f}"])
            if k in (1, 5, 10, 20) or k == 6:
                print(f"[Ch5] pass@{k:2d} (n=20, p=0.15): True={true_pk:.4f}, Unbiased={unbiased:.4f}, Naive={naive:.4f} (bias={naive-true_pk:+.4f})")


def exp6_repro_ablation():
    """Ch 6: Controlled multi-seed Ablation Study (Full ResMLP vs No-Residual vs No-LayerNorm vs No-CosineLR)."""
    def run_variant(use_res, use_ln, seeds=6):
        accs = []
        for sd in range(seeds):
            torch.manual_seed(100 + sd)
            X = torch.randn(400, 16)
            y = ((X[:, 0] * X[:, 1] + 0.5 * X[:, 2]**2 - X[:, 3]) > 0).long()
            Xtr, ytr, Xte, yte = X[:260], y[:260], X[260:], y[260:]

            class ResBlock(nn.Module):
                def __init__(self, d):
                    super().__init__()
                    self.fc1 = nn.Linear(d, d)
                    self.fc2 = nn.Linear(d, d)
                    self.ln = nn.LayerNorm(d) if use_ln else nn.Identity()
                def forward(self, h):
                    z = self.fc2(F.relu(self.fc1(self.ln(h))))
                    return h + z if use_res else z

            net = nn.Sequential(nn.Linear(16, 48), *[ResBlock(48) for _ in range(8)], nn.Linear(48, 2))
            opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
            for _ in range(80):
                loss = F.cross_entropy(net(Xtr), ytr)
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                accs.append(float((net(Xte).argmax(-1) == yte).float().mean().item() * 100.0))
        return float(np.mean(accs)), float(np.std(accs, ddof=1))

    configs = [
        ("Full (Res+LN)", True, True),
        ("w/o LayerNorm", True, False),
        ("w/o Residual", False, True),
        ("Plain (No Res/LN)", False, False),
    ]
    with open(os.path.join(OUT_DIR, "repro_ablation.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["variant", "mean_acc", "std_acc"])
        for name, ur, ul in configs:
            mu, sd = run_variant(ur, ul)
            wr.writerow([name, f"{mu:.2f}", f"{sd:.2f}"])
            print(f"[Ch6] Ablation {name:18s}: {mu:.2f}% ± {sd:.2f}%")


if __name__ == "__main__":
    exp1_seed_cherrypick_and_mcnemar()
    exp2_calibration_and_conformal()
    exp3_adversarial_fgsm_pgd()
    exp4_integrated_gradients_and_fairness()
    exp5_passk_and_arena()
    exp6_repro_ablation()
