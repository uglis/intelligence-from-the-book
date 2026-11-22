"""
Verified Python/NumPy experiments for AI Series Volume 2:
AI Probability, Statistics, Information Theory, and Pure NumPy Linear/Logistic/Softmax Regression.
Generates real CSV files in ../data/ for pgfplots and prints verified metrics.
"""
import os
import csv
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def exp1_bayes_rare_disease():
    """Ch 1: Rare disease Bayes update + sequential 2nd test."""
    prior = 0.001
    sens = 0.99   # P(+ | D)
    spec = 0.98   # P(- | ~D) => FPR = 0.02
    fpr = 1.0 - spec
    post1 = (sens * prior) / (sens * prior + fpr * (1.0 - prior))
    post2 = (sens * post1) / (sens * post1 + fpr * (1.0 - post1))
    print(f"[Ch1] Rare disease: prior={prior:.4f}, after 1st (+): {post1:.5f} ({post1*100:.2f}%), after 2nd (+): {post2:.5f} ({post2*100:.2f}%)")


def exp2_mvg_sampling():
    """Ch 2: Cholesky sampling of 2D correlated Gaussian + empirical conditional check."""
    rng = np.random.default_rng(42)
    mu = np.array([1.0, 2.0])
    Sigma = np.array([[4.0, 1.8], [1.8, 1.0]])
    L = np.linalg.cholesky(Sigma)
    Z = rng.normal(size=(50000, 2))
    X = mu + Z @ L.T
    emp_mu = X.mean(axis=0)
    emp_cov = np.cov(X, rowvar=False)
    # Conditional X1 | X2 = 3.0
    # Exact: mu_{1|2} = 1 + 1.8/1.0 * (3 - 2) = 2.8, var_{1|2} = 4 - 1.8^2/1.0 = 0.76
    mask = np.abs(X[:, 1] - 3.0) < 0.05
    cond_mean = float(X[mask, 0].mean())
    cond_var = float(X[mask, 0].var())
    print(f"[Ch2] MVG emp_mu={np.round(emp_mu,3)}, emp_cov={np.round(emp_cov.ravel(),3)}")
    print(f"[Ch2] X1|X2~3.0 (n={mask.sum()}): emp_mean={cond_mean:.3f} (exact 2.800), emp_var={cond_var:.3f} (exact 0.760)")


def exp3_hoeffding_clt():
    """Ch 3: Hoeffding bound vs Chebyshev vs empirical tail for Bernoulli(0.5) mean."""
    rng = np.random.default_rng(2026)
    eps = 0.1
    trials = 20000
    with open(os.path.join(OUT_DIR, "hoeffding_tail.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "emp_tail", "hoeffding", "chebyshev"])
        for n in [10, 20, 40, 60, 80, 100, 150, 200, 300]:
            means = rng.binomial(n, 0.5, size=trials) / n
            emp = float(np.mean(np.abs(means - 0.5) >= eps))
            hoeff = float(2.0 * np.exp(-2.0 * n * eps * eps))
            cheb = float(min(1.0, 0.25 / (n * eps * eps)))
            w.writerow([n, f"{max(emp, 1e-5):.5f}", f"{hoeff:.5f}", f"{cheb:.5f}"])
            if n in (20, 100, 200, 300):
                print(f"[Ch3] n={n:3d}: emp={emp:.5f}, Hoeffding={hoeff:.5f}, Chebyshev={cheb:.5f}")


def exp4_kl_asymmetry():
    """Ch 4: Forward KL (mode-covering) vs Reverse KL (mode-seeking) on 1D bimodal mixture."""
    x = np.linspace(-6.0, 6.0, 601)
    dx = x[1] - x[0]
    def norm_pdf(z, m, s):
        return np.exp(-0.5 * ((z - m) / s) ** 2) / (np.sqrt(2.0 * np.pi) * s)
    # Bimodal target P(x) = 0.5 N(-2, 0.6^2) + 0.5 N(2, 0.6^2)
    p = 0.5 * norm_pdf(x, -2.0, 0.6) + 0.5 * norm_pdf(x, 2.0, 0.6)
    p /= np.sum(p) * dx
    # Forward KL min over single Gaussian -> exact moment matching: mu=0, var = 0.6^2 + 2^2 = 4.36 -> sigma=2.088
    q_fwd = norm_pdf(x, 0.0, np.sqrt(4.36))
    # Reverse KL min over single Gaussian -> locks onto one mode: mu=2.0, sigma=0.6
    q_rev = norm_pdf(x, 2.0, 0.6)
    kl_pq_fwd = float(np.sum(p * np.log((p + 1e-15) / (q_fwd + 1e-15))) * dx)
    kl_pq_rev = float(np.sum(p * np.log((p + 1e-15) / (q_rev + 1e-15))) * dx)
    kl_qp_fwd = float(np.sum(q_fwd * np.log((q_fwd + 1e-15) / (p + 1e-15))) * dx)
    kl_qp_rev = float(np.sum(q_rev * np.log((q_rev + 1e-15) / (p + 1e-15))) * dx)
    print(f"[Ch4] Bimodal KL: q_fwd(mu=0,s=2.09): KL(P||Q)={kl_pq_fwd:.3f}, KL(Q||P)={kl_qp_fwd:.3f}")
    print(f"[Ch4] Bimodal KL: q_rev(mu=2,s=0.60): KL(P||Q)={kl_pq_rev:.3f}, KL(Q||P)={kl_qp_rev:.3f}")

    with open(os.path.join(OUT_DIR, "kl_bimodal.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "p", "qfwd", "qrev"])
        for i in range(0, len(x), 4):
            w.writerow([f"{x[i]:.2f}", f"{p[i]:.5f}", f"{q_fwd[i]:.5f}", f"{q_rev[i]:.5f}"])


def exp5_mle_map_bootstrap():
    """Ch 5: Beta-Bernoulli MAP vs MLE + Bootstrap 95% CI demo."""
    # Coin flip with n=3, k=3 heads, Beta(2,2) prior -> MAP = (3+1)/(3+2) = 0.8, post mean = (3+2)/(3+4)=5/7
    print(f"[Ch5] 3/3 heads: MLE=1.000, Beta(2,2) MAP={(3+2-1)/(3+2+2-2):.4f}, PostMean={(3+2)/(3+4):.4f}")

    # Bootstrap CI on validation accuracy (153 correct out of 180 -> 85.00%)
    rng = np.random.default_rng(2026)
    hits = np.array([1] * 153 + [0] * 27)
    B = 5000
    boot_accs = rng.choice(hits, size=(B, len(hits)), replace=True).mean(axis=1) * 100.0
    lo, hi = np.percentile(boot_accs, [2.5, 97.5])
    print(f"[Ch5] Bootstrap 95% CI for 153/180 (85.00%): [{lo:.2f}%, {hi:.2f}%], std={boot_accs.std():.2f}%")


def exp6_leakage_and_logreg():
    """Ch 6: (1) Pure-noise feature-selection data leakage experiment; (2) Full 30-dim Logistic L2 sweep + Bootstrap CI."""
    # Part 1: Pure noise N=60, D=500, 50 repeats
    rng_l = np.random.default_rng(77)
    acc_leak, acc_clean = [], []
    for _ in range(50):
        X = rng_l.normal(size=(60, 500))
        y = rng_l.integers(0, 2, size=60)
        # Leak: top-10 correlated features on ALL 60
        corrs_all = np.abs(X.T @ (y - y.mean()))
        top_all = np.argsort(corrs_all)[-10:]
        Xtr, ytr, Xte, yte = X[:40, top_all], y[:40], X[40:, top_all], y[40:]
        w_l, *_ = np.linalg.lstsq(np.column_stack([Xtr, np.ones(40)]), ytr, rcond=None)
        p_l = ((np.column_stack([Xte, np.ones(20)]) @ w_l) >= 0.5).astype(int)
        acc_leak.append(np.mean(p_l == yte) * 100.0)

        # Clean: top-10 correlated features ONLY on train 40
        corrs_tr = np.abs(X[:40].T @ (y[:40] - y[:40].mean()))
        top_tr = np.argsort(corrs_tr)[-10:]
        Xtr2, Xte2 = X[:40, top_tr], X[40:, top_tr]
        w_c, *_ = np.linalg.lstsq(np.column_stack([Xtr2, np.ones(40)]), ytr, rcond=None)
        p_c = ((np.column_stack([Xte2, np.ones(20)]) @ w_c) >= 0.5).astype(int)
        acc_clean.append(np.mean(p_c == yte) * 100.0)
    print(f"[Ch6] Pure-noise leakage (50 runs): leaked_acc={np.mean(acc_leak):.2f}% vs clean_acc={np.mean(acc_clean):.2f}% (true Bayes=50%)")

    # Part 2: 30-feature Logistic Regression L2 sweep (120 train / 180 val)
    rng = np.random.default_rng(2026)
    n_per = 150
    X0 = rng.multivariate_normal([-1.0, -0.8], [[1.3, 0.5], [0.5, 1.2]], size=n_per)
    X1 = rng.multivariate_normal([1.0, 0.8], [[1.2, -0.4], [-0.4, 1.3]], size=n_per)
    scales = rng.uniform(0.05, 80.0, size=28)
    X_noise = rng.normal(0, 1, size=(2 * n_per, 28)) * scales
    X = np.hstack([np.vstack([X0, X1]), X_noise])
    y = np.concatenate([np.zeros(n_per), np.ones(n_per)])

    idx = rng.permutation(len(y))
    X_tr_raw, y_tr = X[idx[:120]], y[idx[:120]]
    X_va_raw, y_va = X[idx[120:]], y[idx[120:]]
    mu, sd = X_tr_raw.mean(axis=0), X_tr_raw.std(axis=0)
    X_tr, X_va = (X_tr_raw - mu) / sd, (X_va_raw - mu) / sd

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))

    def bce(Xa, ya, w, b, lam):
        z = Xa @ w + b
        return float(np.mean(np.logaddexp(0.0, z) - ya * z) + 0.5 * lam * np.dot(w, w))

    def fit(Xa, ya, Xv, yv, lr=0.25, lam=0.0, epochs=800):
        n, d = Xa.shape
        w, b = np.zeros(d), 0.0
        th, vh = [], []
        for ep in range(epochs):
            err = sigmoid(Xa @ w + b) - ya
            w -= lr * ((Xa.T @ err) / n + lam * w)
            b -= lr * float(np.mean(err))
            if ep % 20 == 0:
                th.append((ep, bce(Xa, ya, w, b, 0.0)))
                vh.append((ep, bce(Xv, yv, w, b, 0.0)))
        return w, b, th, vh

    saved = {}
    for lam in [0.0, 0.01, 0.10, 0.50, 2.00, 5.00]:
        w, b, th, vh = fit(X_tr, y_tr, X_va, y_va, lr=0.25, lam=lam, epochs=800)
        p_tr = (sigmoid(X_tr @ w + b) >= 0.5).astype(float)
        p_va = (sigmoid(X_va @ w + b) >= 0.5).astype(float)
        acc_tr = np.mean(p_tr == y_tr) * 100
        acc_va = np.mean(p_va == y_va) * 100
        va_bce = bce(X_va, y_va, w, b, 0.0)
        tp = int(np.sum((p_va == 1) & (y_va == 1)))
        fp = int(np.sum((p_va == 1) & (y_va == 0)))
        fn = int(np.sum((p_va == 0) & (y_va == 1)))
        tn = int(np.sum((p_va == 0) & (y_va == 0)))
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        print(f"[Ch6] lam={lam:4.2f}: ||w||={np.linalg.norm(w):.3f}, tr={acc_tr:.2f}%, va={acc_va:.2f}%, bce={va_bce:.4f}, TP={tp},FP={fp},FN={fn},TN={tn}, F1={f1:.4f}")
        if lam in (0.0, 0.10):
            saved[lam] = (th, vh)

    with open(os.path.join(OUT_DIR, "logreg_curve.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["epoch", "tr0", "va0", "tr01", "va01"])
        for (ep, t0), (_, v0), (_, t1), (_, v1) in zip(
            saved[0.0][0], saved[0.0][1], saved[0.10][0], saved[0.10][1]
        ):
            wr.writerow([ep, f"{t0:.5f}", f"{v0:.5f}", f"{t1:.5f}", f"{v1:.5f}"])


if __name__ == "__main__":
    exp1_bayes_rare_disease()
    exp2_mvg_sampling()
    exp3_hoeffding_clt()
    exp4_kl_asymmetry()
    exp5_mle_map_bootstrap()
    exp6_leakage_and_logreg()
