"""
Verified Python/NumPy/PyTorch experiments for AI Series Volume 6:
Reinforcement Learning, LLM Post-Training & AI Systems.
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


def exp1_cliff_sarsa_vs_q():
    """Ch 1: Classic 4x12 CliffWalking — SARSA (safe online) vs Q-Learning (optimal greedy, risky online)."""
    rng = np.random.default_rng(2026)
    H, W = 4, 12
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # U, D, L, R

    def step(s, a):
        r, c = divmod(s, W)
        dr, dc = actions[a]
        nr, nc = min(max(r + dr, 0), H - 1), min(max(c + dc, 0), W - 1)
        if nr == 3 and 1 <= nc <= 10:
            return 3 * W + 0, -100.0, False  # fell off cliff -> back to start
        ns = nr * W + nc
        if ns == 3 * W + 11:
            return ns, -1.0, True
        return ns, -1.0, False

    def run_algo(mode, episodes=500, eps=0.1, alpha=0.5, gamma=1.0, seeds=20):
        all_ret = np.zeros((seeds, episodes))
        greedy_lens = []
        for sd in range(seeds):
            local_rng = np.random.default_rng(100 + sd)
            Q = np.zeros((H * W, 4))
            def pick(s):
                if local_rng.random() < eps:
                    return int(local_rng.integers(0, 4))
                return int(np.argmax(Q[s]))
            for ep in range(episodes):
                s = 3 * W + 0
                a = pick(s)
                tot = 0.0
                for _ in range(200):
                    ns, rew, done = step(s, a)
                    tot += rew
                    if mode == "sarsa":
                        na = pick(ns)
                        target = rew + (0.0 if done else gamma * Q[ns, na])
                        Q[s, a] += alpha * (target - Q[s, a])
                        s, a = ns, na
                    else:
                        target = rew + (0.0 if done else gamma * np.max(Q[ns]))
                        Q[s, a] += alpha * (target - Q[s, a])
                        s = ns
                        a = pick(s)
                    if done:
                        break
                all_ret[sd, ep] = tot
            # Evaluate greedy policy
            s = 3 * W + 0
            g_ret = 0.0
            for _ in range(50):
                a = int(np.argmax(Q[s]))
                s, rew, done = step(s, a)
                g_ret += rew
                if done:
                    break
            greedy_lens.append(g_ret)
        return all_ret.mean(axis=0), float(np.mean(greedy_lens))

    sarsa_ret, sarsa_greedy = run_algo("sarsa")
    ql_ret, ql_greedy = run_algo("qlearning")

    # Smooth with 10-episode window for clean plot
    with open(os.path.join(OUT_DIR, "rl_cliff.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["ep", "sarsa", "qlearning"])
        for ep in range(10, 501, 20):
            sr = float(np.mean(sarsa_ret[ep-10:ep]))
            qr = float(np.mean(ql_ret[ep-10:ep]))
            wr.writerow([ep, f"{sr:.2f}", f"{qr:.2f}"])
    print(f"[Ch1] CliffWalking last-100 online avg: SARSA={sarsa_ret[-100:].mean():.2f} (greedy={sarsa_greedy:.1f}) | Q-Learning={ql_ret[-100:].mean():.2f} (greedy={ql_greedy:.1f})")


def exp2_pg_baseline_ac():
    """Ch 2: REINFORCE (no baseline) vs REINFORCE+Baseline vs Actor-Critic on a contextual bandit / 1-step-to-horizon control."""
    # Clean 10-step stochastic 1D target-tracking environment with high reward offset (+5) so baseline impact is crystal clear!
    def run_pg(mode, seeds=12, episodes=250):
        curves = np.zeros((seeds, episodes))
        for sd in range(seeds):
            torch.manual_seed(2026 + sd)
            policy = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2))
            val_net = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
            opt_p = torch.optim.Adam(policy.parameters(), lr=1.5e-2)
            opt_v = torch.optim.Adam(val_net.parameters(), lr=3e-2)
            rng = np.random.default_rng(1000 + sd)
            for ep in range(episodes):
                pos = float(rng.uniform(-1.0, 1.0))
                logps, rews, states = [], [], []
                for t in range(12):
                    st = torch.tensor([pos, t / 12.0], dtype=torch.float32)
                    states.append(st)
                    logits = policy(st)
                    dist = torch.distributions.Categorical(logits=logits)
                    a = dist.sample()
                    logps.append(dist.log_prob(a))
                    # Action 0 moves left (-0.25), action 1 moves right (+0.25)
                    pos = float(np.clip(pos + (0.25 if a.item() == 1 else -0.25) + rng.normal(0, 0.04), -1.5, 1.5))
                    # Shifted positive reward (+4.0 baseline offset!) -> huge variance without baseline!
                    r = 4.0 + (1.0 - abs(pos))
                    rews.append(r)
                curves[sd, ep] = sum(rews)
                # Compute returns G_t
                Gs = []
                g = 0.0
                for r in reversed(rews):
                    g = r + 0.95 * g
                    Gs.insert(0, g)
                Gs_t = torch.tensor(Gs, dtype=torch.float32)
                logps_t = torch.stack(logps)
                states_t = torch.stack(states)

                if mode == "reinforce":
                    loss_p = -(logps_t * Gs_t).mean()
                    opt_p.zero_grad(); loss_p.backward(); opt_p.step()
                elif mode == "baseline":
                    vals = val_net(states_t).squeeze(-1)
                    adv = Gs_t - vals.detach()
                    loss_p = -(logps_t * adv).mean()
                    loss_v = F.mse_loss(vals, Gs_t)
                    opt_p.zero_grad(); loss_p.backward(); opt_p.step()
                    opt_v.zero_grad(); loss_v.backward(); opt_v.step()
                else:  # actor-critic with GAE(lambda=0.9)
                    vals = val_net(states_t).squeeze(-1)
                    rews_t = torch.tensor(rews, dtype=torch.float32)
                    next_vals = torch.cat([vals[1:].detach(), torch.zeros(1)])
                    deltas = rews_t + 0.95 * next_vals - vals.detach()
                    advs = []
                    gae = 0.0
                    for d in reversed(deltas.tolist()):
                        gae = d + 0.95 * 0.90 * gae
                        advs.insert(0, gae)
                    adv = torch.tensor(advs, dtype=torch.float32)
                    loss_p = -(logps_t * adv).mean()
                    loss_v = F.mse_loss(vals, adv + vals.detach())
                    opt_p.zero_grad(); loss_p.backward(); opt_p.step()
                    opt_v.zero_grad(); loss_v.backward(); opt_v.step()
        return curves.mean(axis=0), curves.std(axis=0)

    r_mean, r_std = run_pg("reinforce")
    b_mean, b_std = run_pg("baseline")
    ac_mean, ac_std = run_pg("ac")

    with open(os.path.join(OUT_DIR, "pg_baseline.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["ep", "reinforce", "baseline", "ac"])
        for ep in range(10, 251, 15):
            wr.writerow([ep, f"{r_mean[ep-10:ep].mean():.3f}", f"{b_mean[ep-10:ep].mean():.3f}", f"{ac_mean[ep-10:ep].mean():.3f}"])
    print(f"[Ch2] Final-30 return: REINFORCE={r_mean[-30:].mean():.2f} (±{r_std[-30:].mean():.2f}), +Baseline={b_mean[-30:].mean():.2f} (±{b_std[-30:].mean():.2f}), AC={ac_mean[-30:].mean():.2f} (±{ac_std[-30:].mean():.2f})")


def exp3_ppo_vs_unclipped():
    """Ch 3: Multi-epoch (K=10) Unclipped IS Policy Gradient (collapses!) vs PPO-Clip (stable) on 1D continuous Gaussian LQR control."""
    def run_multi_epoch(clip_eps):
        torch.manual_seed(2026)
        rng = np.random.default_rng(2026)
        mean_net = nn.Sequential(nn.Linear(2, 24), nn.Tanh(), nn.Linear(24, 1))
        log_std = nn.Parameter(torch.tensor([-0.5]))
        val_net = nn.Sequential(nn.Linear(2, 24), nn.Tanh(), nn.Linear(24, 1))
        opt = torch.optim.Adam(list(mean_net.parameters()) + [log_std] + list(val_net.parameters()), lr=4e-2)
        rets = []
        for it in range(50):
            S_buf, A_buf, LP_buf, G_buf = [], [], [], []
            ep_rets = []
            for _ in range(16):
                pos = float(rng.uniform(-1.5, 1.5))
                ep_r = []
                for t in range(10):
                    st = torch.tensor([pos, t / 10.0], dtype=torch.float32)
                    with torch.no_grad():
                        std = torch.exp(torch.clamp(log_std, -2.0, 1.0))
                        dist = torch.distributions.Normal(mean_net(st), std)
                        a = dist.sample()
                        lp = dist.log_prob(a).sum()
                    S_buf.append(st); A_buf.append(a); LP_buf.append(lp)
                    act = float(np.clip(a.item(), -2.0, 2.0))
                    pos = float(np.clip(pos + 0.5 * act + rng.normal(0, 0.05), -3.0, 3.0))
                    ep_r.append(-(pos ** 2) - 0.1 * (act ** 2))
                ep_rets.append(sum(ep_r))
                g = 0.0
                gs = []
                for r in reversed(ep_r):
                    g = r + 0.95 * g
                    gs.insert(0, g)
                G_buf.extend(gs)
            rets.append(float(np.mean(ep_rets)))
            S_t = torch.stack(S_buf)
            A_t = torch.stack(A_buf)
            LP_old = torch.stack(LP_buf)
            G_t = torch.tensor(G_buf, dtype=torch.float32)
            with torch.no_grad():
                adv = G_t - val_net(S_t).squeeze(-1)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            for _ in range(10):
                std = torch.exp(torch.clamp(log_std, -2.0, 1.0))
                dist = torch.distributions.Normal(mean_net(S_t), std)
                lp_new = dist.log_prob(A_t).sum(dim=-1)
                ratio = torch.exp(torch.clamp(lp_new - LP_old, -10.0, 10.0))
                if clip_eps is None:
                    surr = ratio * adv
                else:
                    surr = torch.min(ratio * adv, torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv)
                v_loss = F.mse_loss(val_net(S_t).squeeze(-1), G_t)
                loss = -surr.mean() + 0.5 * v_loss
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(list(mean_net.parameters()) + [log_std], 2.0)
                opt.step()
        return rets

    r_unclip = run_multi_epoch(clip_eps=None)
    r_ppo = run_multi_epoch(clip_eps=0.2)

    with open(os.path.join(OUT_DIR, "ppo_vs_unclip.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["iter", "unclipped", "ppo"])
        for it in range(0, 50, 2):
            wr.writerow([it + 1, f"{r_unclip[it]:.3f}", f"{r_ppo[it]:.3f}"])
    print(f"[Ch3] Continuous LQR Multi-epoch (K=10): iter 1 ({r_ppo[0]:.2f}) -> last-10 avg: Unclipped={np.mean(r_unclip[-10:]):.2f} (worst={min(r_unclip[5:]):.2f}), PPO-Clip={np.mean(r_ppo[-10:]):.2f}")


def exp4_dpo_alignment():
    """Ch 4: Direct Preference Optimization (DPO) on contextual discrete response preference pairs."""
    torch.manual_seed(2026)
    N_ctx, K_resp = 64, 8
    X = torch.randn(N_ctx, 12)
    # True hidden human reward function r*(x, y)
    W_true = torch.randn(12, K_resp)
    R_true = X @ W_true  # (N_ctx, K_resp)

    # Reference policy pi_ref (imperfect SFT policy)
    ref_net = nn.Linear(12, K_resp)
    with torch.no_grad():
        ref_net.weight.copy_(0.25 * W_true.T + 0.6 * torch.randn_like(ref_net.weight))
        ref_net.bias.zero_()
    ref_logits = ref_net(X).detach()
    ref_logp = F.log_softmax(ref_logits, dim=-1)

    # Sample preference pairs (y_w, y_l) from Bradley-Terry model
    pairs_x, pairs_w, pairs_l = [], [], []
    rng = np.random.default_rng(2026)
    for i in range(N_ctx):
        for _ in range(6):
            y1, y2 = rng.choice(K_resp, size=2, replace=False)
            p_1_wins = 1.0 / (1.0 + math.exp(-(R_true[i, y1].item() - R_true[i, y2].item())))
            if rng.random() < p_1_wins:
                yw, yl = y1, y2
            else:
                yw, yl = y2, y1
            pairs_x.append(i); pairs_w.append(yw); pairs_l.append(yl)
    px = torch.tensor(pairs_x, dtype=torch.long)
    pw = torch.tensor(pairs_w, dtype=torch.long)
    pl = torch.tensor(pairs_l, dtype=torch.long)

    def train_dpo(beta):
        pi_net = nn.Linear(12, K_resp)
        pi_net.load_state_dict(ref_net.state_dict())
        opt = torch.optim.Adam(pi_net.parameters(), lr=3e-2)
        hist = []
        for ep in range(121):
            logp = F.log_softmax(pi_net(X), dim=-1)
            dw = logp[px, pw] - ref_logp[px, pw]
            dl = logp[px, pl] - ref_logp[px, pl]
            margin = beta * (dw - dl)
            loss = -F.logsigmoid(margin).mean()
            if ep % 10 == 0:
                with torch.no_grad():
                    probs = torch.exp(logp)
                    exp_rew = float((probs * R_true).sum(dim=-1).mean().item())
                    kl = float((probs * (logp - ref_logp)).sum(dim=-1).mean().item())
                    win_acc = float((margin > 0).float().mean().item() * 100.0)
                hist.append((ep, loss.item(), exp_rew, kl, win_acc))
            opt.zero_grad(); loss.backward(); opt.step()
        return hist

    h_low = train_dpo(beta=0.1)
    h_high = train_dpo(beta=0.6)
    with open(os.path.join(OUT_DIR, "dpo_align.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["ep", "rew_b01", "kl_b01", "rew_b06", "kl_b06", "acc_b01"])
        for a, b in zip(h_low, h_high):
            wr.writerow([a[0], f"{a[2]:.4f}", f"{a[3]:.4f}", f"{b[2]:.4f}", f"{b[3]:.4f}", f"{a[4]:.2f}"])
    print(f"[Ch4] DPO ep=0: rew={h_low[0][2]:.3f} -> ep=120: beta=0.1 (rew={h_low[-1][2]:.3f}, KL={h_low[-1][3]:.3f}, acc={h_low[-1][4]:.1f}%) | beta=0.6 (rew={h_high[-1][2]:.3f}, KL={h_high[-1][3]:.3f}, acc={h_high[-1][4]:.1f}%)")


def exp5_flash_online_and_roofline():
    """Ch 5: Exact Tiled Online Softmax Attention (FlashAttention core) verification + Roofline arithmetic intensity."""
    rng = np.random.default_rng(2026)
    N, d = 256, 64
    Q = rng.normal(size=(N, d))
    K = rng.normal(size=(N, d))
    V = rng.normal(size=(N, d))
    scale = 1.0 / math.sqrt(d)

    # Standard Attention (materializes full N x N score matrix)
    S = (Q @ K.T) * scale
    S_shift = S - np.max(S, axis=-1, keepdims=True)
    P = np.exp(S_shift)
    P /= np.sum(P, axis=-1, keepdims=True)
    O_std = P @ V

    # Tiled Online Softmax Attention (FlashAttention math: never materializes full N x N!)
    Br, Bc = 32, 32
    O_flash = np.zeros_like(Q)
    for i in range(0, N, Br):
        Qi = Q[i:i+Br]
        mi = np.full((Br, 1), -np.inf)
        li = np.zeros((Br, 1))
        Oi = np.zeros((Br, d))
        for j in range(0, N, Bc):
            Kj, Vj = K[j:j+Bc], V[j:j+Bc]
            Sij = (Qi @ Kj.T) * scale
            mij = np.max(Sij, axis=-1, keepdims=True)
            m_new = np.maximum(mi, mij)
            alpha = np.exp(mi - m_new)
            pij = np.exp(Sij - m_new)
            l_new = alpha * li + np.sum(pij, axis=-1, keepdims=True)
            Oi = (Oi * (alpha * li) + pij @ Vj) / l_new
            mi, li = m_new, l_new
        O_flash[i:i+Br] = Oi

    max_err = float(np.max(np.abs(O_std - O_flash)))
    assert max_err < 1e-12
    print(f"[Ch5] FlashAttention Online Softmax vs Std Attention (N={N}, Br=Bc={Br}): max_err={max_err:.2e}, peak_tile_ratio={(Br*Bc)/(N*N)*100:.2f}%")

    # Roofline analysis for a 4096x4096 BF16 linear layer on an H100-class GPU (989 TFLOP/s, 3.35 TB/s -> ridge=295 FLOP/B)
    peak_tflops = 989.0
    bw_tbs = 3.35
    d_model = 4096
    with open(os.path.join(OUT_DIR, "roofline.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["tokens", "ai", "achieved_tflops", "util_pct"])
        for T_tok in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            flops = 2.0 * T_tok * d_model * d_model
            bytes_rw = 2.0 * (d_model * d_model + 2.0 * T_tok * d_model)  # BF16 = 2 bytes
            ai = flops / bytes_rw
            achieved = min(peak_tflops, ai * bw_tbs)
            wr.writerow([T_tok, f"{ai:.2f}", f"{achieved:.2f}", f"{achieved/peak_tflops*100:.2f}"])
    print(f"[Ch5] Roofline 4096x4096 BF16 (ridge=295 FLOP/B): T=1 -> AI=1.00 (3.35 TFLOP/s, 0.3%), T=512 -> AI=409.6 (989 TFLOP/s, 100%)")


def exp6_quant_and_lora():
    """Ch 6: Per-tensor vs Per-channel INT8/INT4 Quantization MSE + LoRA rank r sweep."""
    rng = np.random.default_rng(2026)
    # Create weight matrix with realistic heavy-tailed channel outliers
    W = rng.normal(0, 0.25, size=(256, 256))
    W[0:4, :] *= 8.0  # 4 outlier channels!
    X = rng.normal(0, 1.0, size=(256, 64))
    Y_fp = W @ X

    def quant_dequant(M, bits, per_channel=False):
        qmax = (1 << (bits - 1)) - 1
        axis = 1 if per_channel else None
        scale = np.max(np.abs(M), axis=axis, keepdims=True) / qmax
        scale = np.maximum(scale, 1e-8)
        Q = np.clip(np.round(M / scale), -qmax, qmax)
        return Q * scale

    for bits in [8, 4]:
        W_pt = quant_dequant(W, bits, per_channel=False)
        W_pc = quant_dequant(W, bits, per_channel=True)
        rel_pt = float(np.linalg.norm(W_pt @ X - Y_fp) / np.linalg.norm(Y_fp) * 100.0)
        rel_pc = float(np.linalg.norm(W_pc @ X - Y_fp) / np.linalg.norm(Y_fp) * 100.0)
        print(f"[Ch6] INT{bits} Quantization output rel-error: per-tensor={rel_pt:.3f}%, per-channel={rel_pc:.3f}%")

    # LoRA low-rank adaptation sweep on a frozen 64x64 linear layer adapting to a rank-4 task shift
    torch.manual_seed(2026)
    D = 64
    W0 = torch.randn(D, D) * 0.2
    U_t, V_t = torch.randn(D, 4) * 0.4, torch.randn(4, D) * 0.4
    W_target = W0 + U_t @ V_t + 0.03 * torch.randn(D, D)
    X_tr = torch.randn(256, D)
    Y_tr = X_tr @ W_target.T
    X_te = torch.randn(128, D)
    Y_te = X_te @ W_target.T

    with open(os.path.join(OUT_DIR, "lora_rank.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["rank", "params", "param_pct", "test_mse"])
        for r in [1, 2, 4, 8, 16, 64]:
            if r < 64:
                A = nn.Parameter(torch.randn(r, D) * 0.05)
                B = nn.Parameter(torch.zeros(D, r))
                opt = torch.optim.Adam([A, B], lr=5e-2)
                for _ in range(250):
                    pred = X_tr @ (W0 + B @ A).T
                    loss = F.mse_loss(pred, Y_tr)
                    opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    te_mse = float(F.mse_loss(X_te @ (W0 + B @ A).T, Y_te).item())
                n_params = 2 * D * r
            else:
                W_ft = nn.Parameter(W0.clone())
                opt = torch.optim.Adam([W_ft], lr=5e-2)
                for _ in range(250):
                    loss = F.mse_loss(X_tr @ W_ft.T, Y_tr)
                    opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    te_mse = float(F.mse_loss(X_te @ W_ft.T, Y_te).item())
                n_params = D * D
            wr.writerow([r, n_params, f"{n_params/(D*D)*100:.2f}", f"{te_mse:.5f}"])
            if r in (1, 4, 8, 64):
                print(f"[Ch6] LoRA rank={r:2d}: params={n_params:4d} ({n_params/(D*D)*100:5.1f}%), test_mse={te_mse:.5f}")


if __name__ == "__main__":
    exp1_cliff_sarsa_vs_q()
    exp2_pg_baseline_ac()
    exp3_ppo_vs_unclipped()
    exp4_dpo_alignment()
    exp5_flash_online_and_roofline()
    exp6_quant_and_lora()
