"""
Verified Python/NumPy experiments for AI Series Volume 4:
Convex Optimization & Classical Machine Learning.
Generates real CSV files in ../data/ for pgfplots and prints verified metrics.
"""
import os
import csv
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def exp1_opt_acceleration():
    """Ch 1: GD vs Heavy-Ball Momentum vs Nesterov AGD on ill-conditioned L2 Logistic Regression."""
    rng = np.random.default_rng(2026)
    N, D = 300, 15
    scales = np.geomspace(1.0, 15.0, D)
    X = rng.normal(size=(N, D)) * scales
    w_true = rng.normal(size=D) / scales
    probs = 1.0 / (1.0 + np.exp(-X @ w_true))
    y = (rng.random(N) < probs).astype(float)
    lam = 0.05

    # Exact smoothness upper bound L = 0.25/N * lambda_max(X^T X) + lam
    smax = np.linalg.svd(X, compute_uv=False)[0]
    L = 0.25 * (smax ** 2) / N + lam
    mu = lam
    kappa = L / mu
    eta = 1.0 / L

    def loss_grad(w):
        z = X @ w
        loss = float(np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * lam * np.dot(w, w))
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        grad = (X.T @ (p - y)) / N + lam * w
        return loss, grad

    # Reference optimum via Newton's method (15 steps -> machine precision)
    w_star = np.zeros(D)
    for _ in range(15):
        z = X @ w_star
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        g = (X.T @ (p - y)) / N + lam * w_star
        H = (X.T * (p * (1.0 - p))) @ X / N + lam * np.eye(D)
        w_star -= np.linalg.solve(H, g)
    f_star, g_star = loss_grad(w_star)
    assert np.linalg.norm(g_star) < 1e-12

    # Run GD, Momentum (beta=0.9), Nesterov AGD (optimal beta = (sqrt(kappa)-1)/(sqrt(kappa)+1))
    beta_nag = (np.sqrt(kappa) - 1.0) / (np.sqrt(kappa) + 1.0)
    w_gd = np.zeros(D)
    w_mom = np.zeros(D); v_mom = np.zeros(D)
    w_nag = np.zeros(D); y_nag = np.zeros(D)

    with open(os.path.join(OUT_DIR, "opt_accel.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t", "gd", "mom", "nag"])
        for t in range(151):
            l_gd, g_gd = loss_grad(w_gd)
            l_mom, g_mom = loss_grad(w_mom)
            l_nag, _ = loss_grad(w_nag)
            if t % 5 == 0:
                wr.writerow([t, f"{max(l_gd-f_star, 1e-13):.3e}", f"{max(l_mom-f_star, 1e-13):.3e}", f"{max(l_nag-f_star, 1e-13):.3e}"])
            if t in (0, 50, 100, 150):
                print(f"[Ch1] t={t:3d}: GD={l_gd-f_star:.2e}, Mom={l_mom-f_star:.2e}, NAG={l_nag-f_star:.2e}")
            # Updates
            w_gd -= eta * g_gd
            v_mom = 0.85 * v_mom + g_mom
            w_mom -= eta * v_mom
            _, g_y = loss_grad(y_nag)
            w_next = y_nag - eta * g_y
            y_nag = w_next + beta_nag * (w_next - w_nag)
            w_nag = w_next
    print(f"[Ch1] Logistic L={L:.2f}, mu={mu:.2f}, kappa={kappa:.1f}, beta_nag={beta_nag:.3f}")


def exp2_lasso_ista():
    """Ch 2: ISTA (Proximal Gradient) Lasso Path on 20-dim sparse regression (4 true nonzeros)."""
    rng = np.random.default_rng(42)
    N, D = 120, 20
    X = rng.normal(size=(N, D))
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    w_true = np.zeros(D)
    w_true[0], w_true[1], w_true[2], w_true[3] = 3.5, -2.5, 1.8, -1.2
    y = X @ w_true + rng.normal(0, 0.5, size=N)
    L = float(np.linalg.svd(X, compute_uv=False)[0] ** 2 / N)
    eta = 1.0 / L

    def soft_thresh(z, tau):
        return np.sign(z) * np.maximum(np.abs(z) - tau, 0.0)

    lams = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.9, 1.2, 1.6, 2.2, 3.0, 3.8]
    with open(os.path.join(OUT_DIR, "lasso_path.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["lam", "w0", "w1", "w2", "w3", "w4_noise", "nnz"])
        for lam in lams:
            w = np.zeros(D)
            for _ in range(1200):
                grad = (X.T @ (X @ w - y)) / N
                w = soft_thresh(w - eta * grad, eta * lam)
            nnz = int(np.sum(np.abs(w) > 1e-4))
            wr.writerow([f"{lam:.2f}", f"{w[0]:.4f}", f"{w[1]:.4f}", f"{w[2]:.4f}", f"{w[3]:.4f}", f"{w[4]:.4f}", nnz])
            if lam in (0.0, 0.2, 0.6, 1.2, 3.0):
                print(f"[Ch2] Lasso lam={lam:.2f}: nnz={nnz:2d}, w[:5]={np.round(w[:5], 3)}")


def exp3_bias_variance():
    """Ch 3: Exact empirical Bias^2 + Variance + Noise decomposition on noisy sin(2*pi*x)."""
    rng = np.random.default_rng(7)
    x_te = np.linspace(0.05, 0.95, 50)
    f_te = np.sin(2.0 * np.pi * x_te)
    sigma = 0.30
    n_trials = 400
    n_train = 22

    def vander(x, d):
        return np.column_stack([x ** k for k in range(d + 1)])

    with open(os.path.join(OUT_DIR, "bias_var.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["deg", "bias2", "var", "noise", "total"])
        x_grid = np.linspace(0.02, 0.98, n_train)
        for d in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            preds = np.zeros((n_trials, len(x_te)))
            A_te = vander(x_te, d)
            for k in range(n_trials):
                x_tr = np.clip(x_grid + rng.uniform(-0.02, 0.02, size=n_train), 0.0, 1.0)
                y_tr = np.sin(2.0 * np.pi * x_tr) + rng.normal(0.0, sigma, size=n_train)
                A_tr = vander(x_tr, d)
                w, *_ = np.linalg.lstsq(A_tr, y_tr, rcond=None)
                preds[k] = A_te @ w
            mean_pred = preds.mean(axis=0)
            bias2 = float(np.mean((mean_pred - f_te) ** 2))
            var = float(np.mean(np.var(preds, axis=0)))
            noise = sigma ** 2
            total = bias2 + var + noise
            wr.writerow([d, f"{bias2:.5f}", f"{var:.5f}", f"{noise:.5f}", f"{total:.5f}"])
            if d in (1, 3, 5, 7, 9):
                print(f"[Ch3] deg={d}: Bias^2={bias2:.4f}, Var={var:.4f}, Noise={noise:.4f}, Total={total:.4f}")


def exp4_rbf_svm_circles():
    """Ch 4: Pure NumPy Dual Kernel SVM (SMO) on noisy concentric circles."""
    rng = np.random.default_rng(2026)
    n_per = 120
    # Inner circle r ~ 1.0, outer ring r ~ 2.0 with realistic radial overlap
    th0 = rng.uniform(0, 2 * np.pi, n_per)
    r0 = 1.0 + rng.normal(0, 0.28, n_per)
    X0 = np.column_stack([r0 * np.cos(th0), r0 * np.sin(th0)])
    th1 = rng.uniform(0, 2 * np.pi, n_per)
    r1 = 2.0 + rng.normal(0, 0.30, n_per)
    X1 = np.column_stack([r1 * np.cos(th1), r1 * np.sin(th1)])
    X = np.vstack([X0, X1])
    y = np.concatenate([-np.ones(n_per), np.ones(n_per)])
    idx = rng.permutation(len(y))
    Xtr, ytr = X[idx[:160]], y[idx[:160]]
    Xte, yte = X[idx[160:]], y[idx[160:]]

    # 1. Linear SVM (hinge loss subgradient)
    w_lin, b_lin = np.zeros(2), 0.0
    for ep in range(500):
        margins = ytr * (Xtr @ w_lin + b_lin)
        mask = margins < 1.0
        gw = w_lin - 1.0 * (Xtr[mask].T @ ytr[mask]) / len(ytr)
        gb = -1.0 * np.sum(ytr[mask]) / len(ytr)
        w_lin -= 0.05 * gw; b_lin -= 0.05 * gb
    lin_acc = float(np.mean(np.sign(Xte @ w_lin + b_lin) == yte) * 100.0)

    # 2. Dual RBF-SVM via simple SMO coordinate ascent
    def rbf_kernel(A, B, gamma=0.5):
        d2 = np.sum(A**2, axis=1, keepdims=True) + np.sum(B**2, axis=1) - 2.0 * (A @ B.T)
        return np.exp(-gamma * np.maximum(d2, 0.0))

    for gamma in [0.1, 0.8, 12.0]:
        C = 5.0
        K = rbf_kernel(Xtr, Xtr, gamma=gamma)
        n = len(ytr)
        alpha = np.zeros(n)
        b = 0.0
        for _ in range(35):
            for i in range(n):
                fi = np.dot(alpha * ytr, K[:, i]) + b
                Ei = fi - ytr[i]
                if (ytr[i] * Ei < -1e-3 and alpha[i] < C) or (ytr[i] * Ei > 1e-3 and alpha[i] > 0):
                    j = (i + 1 + int(rng.integers(0, n - 1))) % n
                    fj = np.dot(alpha * ytr, K[:, j]) + b
                    Ej = fj - ytr[j]
                    ai_old, aj_old = alpha[i], alpha[j]
                    if ytr[i] != ytr[j]:
                        L_b, H_b = max(0.0, aj_old - ai_old), min(C, C + aj_old - ai_old)
                    else:
                        L_b, H_b = max(0.0, ai_old + aj_old - C), min(C, ai_old + aj_old)
                    if L_b == H_b:
                        continue
                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    alpha[j] = np.clip(aj_old - ytr[j] * (Ei - Ej) / eta, L_b, H_b)
                    if abs(alpha[j] - aj_old) < 1e-5:
                        continue
                    alpha[i] = ai_old + ytr[i] * ytr[j] * (aj_old - alpha[j])
                    b1 = b - Ei - ytr[i] * (alpha[i] - ai_old) * K[i, i] - ytr[j] * (alpha[j] - aj_old) * K[i, j]
                    b2 = b - Ej - ytr[i] * (alpha[i] - ai_old) * K[i, j] - ytr[j] * (alpha[j] - aj_old) * K[j, j]
                    if 0 < alpha[i] < C:
                        b = b1
                    elif 0 < alpha[j] < C:
                        b = b2
                    else:
                        b = 0.5 * (b1 + b2)
        K_te = rbf_kernel(Xte, Xtr, gamma=gamma)
        pred_tr = np.sign(K @ (alpha * ytr) + b)
        pred_te = np.sign(K_te @ (alpha * ytr) + b)
        n_sv = int(np.sum(alpha > 1e-3))
        print(f"[Ch4] Circles RBF-SVM (gamma={gamma:4.1f}): SVs={n_sv:3d}/{n}, tr_acc={np.mean(pred_tr==ytr)*100:.2f}%, te_acc={np.mean(pred_te==yte)*100:.2f}%")
    print(f"[Ch4] Circles Linear SVM te_acc={lin_acc:.2f}%")


def exp5_trees_rf_gbdt():
    """Ch 5: Pure NumPy Regression Tree, Random Forest (Bagging), and GBDT on nonlinear regression."""
    rng = np.random.default_rng(2026)
    N_tr, N_te = 220, 220
    X_tr = rng.uniform(-3.0, 3.0, size=(N_tr, 5))
    f_true = lambda X: np.sin(X[:, 0]) + 0.5 * (X[:, 1] ** 2) - 0.8 * X[:, 2] + 0.4 * X[:, 0] * X[:, 1]
    y_tr = f_true(X_tr) + rng.normal(0.0, 0.70, size=N_tr)
    X_te = rng.uniform(-3.0, 3.0, size=(N_te, 5))
    y_te = f_true(X_te)

    def fit_reg_tree(X, y, max_depth=3, min_leaf=4, max_features=None, rng_local=None):
        n, d = X.shape
        if max_depth == 0 or n <= min_leaf or np.var(y) < 1e-10:
            return float(np.mean(y))
        feats = range(d) if max_features is None else rng_local.choice(d, size=max_features, replace=False)
        best_score, best_split = float("inf"), None
        for j in feats:
            vals = np.unique(X[:, j])
            if len(vals) <= 1:
                continue
            qs = vals if len(vals) <= 16 else np.quantile(vals, np.linspace(0.1, 0.9, 15))
            for thr in qs:
                left = X[:, j] <= thr
                nl = int(np.sum(left))
                if nl < min_leaf or (n - nl) < min_leaf:
                    continue
                yl, yr = y[left], y[~left]
                score = nl * np.var(yl) + (n - nl) * np.var(yr)
                if score < best_score:
                    best_score = score
                    best_split = (j, float(thr))
        if best_split is None:
            return float(np.mean(y))
        j, thr = best_split
        left = X[:, j] <= thr
        return (
            j, thr,
            fit_reg_tree(X[left], y[left], max_depth - 1, min_leaf, max_features, rng_local),
            fit_reg_tree(X[~left], y[~left], max_depth - 1, min_leaf, max_features, rng_local),
        )

    def pred_tree(tree, X):
        if isinstance(tree, float):
            return np.full(len(X), tree)
        j, thr, tl, tr = tree
        out = np.empty(len(X))
        left = X[:, j] <= thr
        out[left] = pred_tree(tl, X[left])
        out[~left] = pred_tree(tr, X[~left])
        return out

    # Single deep tree (high variance)
    deep_tree = fit_reg_tree(X_tr, y_tr, max_depth=8, min_leaf=2)
    single_tr = float(np.mean((pred_tree(deep_tree, X_tr) - y_tr) ** 2))
    single_te = float(np.mean((pred_tree(deep_tree, X_te) - y_te) ** 2))
    print(f"[Ch5] Single deep CART (d=8): tr_mse={single_tr:.4f}, te_mse={single_te:.4f}")

    # Random Forest (60 trees, depth=8, max_features=3) & GBDT (60 trees, depth=3, lr=0.10)
    M_max = 60
    rf_preds = np.zeros(N_te)
    gb_tr = np.full(N_tr, np.mean(y_tr))
    gb_te = np.full(N_te, np.mean(y_tr))
    lr_gb = 0.10

    with open(os.path.join(OUT_DIR, "ensemble_curve.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["m", "single_te", "rf_te", "gbdt_te"])
        for m in range(1, M_max + 1):
            b_idx = rng.integers(0, N_tr, size=N_tr)
            t_rf = fit_reg_tree(X_tr[b_idx], y_tr[b_idx], max_depth=8, min_leaf=2, max_features=3, rng_local=rng)
            rf_preds += pred_tree(t_rf, X_te)
            rf_mse = float(np.mean((rf_preds / m - y_te) ** 2))

            resid = y_tr - gb_tr
            t_gb = fit_reg_tree(X_tr, resid, max_depth=3, min_leaf=4)
            gb_tr += lr_gb * pred_tree(t_gb, X_tr)
            gb_te += lr_gb * pred_tree(t_gb, X_te)
            gb_mse = float(np.mean((gb_te - y_te) ** 2))

            if m in (1, 2, 5, 10, 15, 20, 30, 40, 50, 60):
                wr.writerow([m, f"{single_te:.5f}", f"{rf_mse:.5f}", f"{gb_mse:.5f}"])
            if m in (1, 10, 30, 60):
                print(f"[Ch5] M={m:2d}: RF te_mse={rf_mse:.4f}, GBDT te_mse={gb_mse:.4f}")


def exp6_kmeans_vs_gmm_em():
    """Ch 6: Tilted elliptical clusters — K-Means failure vs GMM-EM monotonic log-likelihood."""
    rng = np.random.default_rng(2026)
    n_per = 180
    # Two elongated parallel tilted Gaussians (classic case where spherical K-Means cuts across both!)
    C_tilt = np.array([[4.0, 2.8], [2.8, 2.2]])
    X0 = rng.multivariate_normal([-1.0, 1.2], C_tilt, size=n_per)
    X1 = rng.multivariate_normal([1.2, -1.0], C_tilt, size=n_per)
    X = np.vstack([X0, X1])
    y_true = np.concatenate([np.zeros(n_per, dtype=int), np.ones(n_per, dtype=int)])

    # 1. K-Means (5 restarts, best inertia)
    best_km_acc = 0.0
    best_inertia = float("inf")
    for _ in range(5):
        cent = X[rng.choice(len(X), size=2, replace=False)].copy()
        for _ in range(50):
            d2 = np.sum((X[:, None, :] - cent[None, :, :]) ** 2, axis=2)
            lbl = np.argmin(d2, axis=1)
            for k in range(2):
                if np.any(lbl == k):
                    cent[k] = X[lbl == k].mean(axis=0)
        inertia = float(np.sum(np.min(d2, axis=1)))
        if inertia < best_inertia:
            best_inertia = inertia
            best_km_acc = max(np.mean(lbl == y_true), np.mean(lbl != y_true)) * 100.0

    # 2. GMM via EM (explicit log-likelihood tracking)
    N, D = X.shape
    K = 2
    pis = np.array([0.5, 0.5])
    mus = X[rng.choice(N, size=K, replace=False)].copy()
    Sigmas = np.array([np.cov(X, rowvar=False) for _ in range(K)])

    def log_gauss(X_in, mu, Sigma):
        diff = X_in - mu
        sign, logdet = np.linalg.slogdet(Sigma)
        invS = np.linalg.inv(Sigma)
        quad = np.sum((diff @ invS) * diff, axis=1)
        return -0.5 * (D * np.log(2.0 * np.pi) + logdet + quad)

    with open(os.path.join(OUT_DIR, "gmm_em.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["iter", "loglik"])
        prev_ll = -float("inf")
        for it in range(1, 31):
            # E-step
            log_prob = np.column_stack([np.log(pis[k] + 1e-12) + log_gauss(X, mus[k], Sigmas[k]) for k in range(K)])
            lse = np.max(log_prob, axis=1, keepdims=True)
            log_norm = lse + np.log(np.sum(np.exp(log_prob - lse), axis=1, keepdims=True))
            ll = float(np.sum(log_norm))
            assert ll >= prev_ll - 1e-7, f"LL decreased: {prev_ll} -> {ll}"
            prev_ll = ll
            wr.writerow([it, f"{ll:.4f}"])
            gamma = np.exp(log_prob - log_norm)
            # M-step
            Nk = gamma.sum(axis=0)
            pis = Nk / N
            for k in range(K):
                mus[k] = (gamma[:, k] @ X) / Nk[k]
                diff = X - mus[k]
                Sigmas[k] = (diff.T * gamma[:, k]) @ diff / Nk[k] + 1e-6 * np.eye(D)

    gmm_lbl = np.argmax(gamma, axis=1)
    gmm_acc = max(np.mean(gmm_lbl == y_true), np.mean(gmm_lbl != y_true)) * 100.0
    print(f"[Ch6] Tilted clusters (N=360): K-Means purity={best_km_acc:.2f}% vs GMM-EM purity={gmm_acc:.2f}%, final LL={prev_ll:.2f}")


if __name__ == "__main__":
    exp1_opt_acceleration()
    exp2_lasso_ista()
    exp3_bias_variance()
    exp4_rbf_svm_circles()
    exp5_trees_rf_gbdt()
    exp6_kmeans_vs_gmm_em()
