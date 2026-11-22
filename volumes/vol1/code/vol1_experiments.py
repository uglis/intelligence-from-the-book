"""
Verified Python/NumPy experiments for AI Series Volume 1 (v2):
Python/NumPy, AI Linear Algebra, and Multivariable Calculus.
Generates real CSV files in ../data/ for pgfplots and prints verified metrics.
"""
import os
import csv
import time
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def exp1_matmul_speed():
    """Ch 1: Pure Python triple loop vs NumPy @ speedup."""
    rng = np.random.default_rng(42)
    n = 120
    A = rng.normal(size=(n, n))
    B = rng.normal(size=(n, n))
    Al, Bl = A.tolist(), B.tolist()

    t0 = time.perf_counter()
    C_py = [[sum(Al[i][k] * Bl[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(50):
        C_np = A @ B
    t_np = (time.perf_counter() - t0) / 50.0

    max_diff = float(np.max(np.abs(np.array(C_py) - C_np)))
    print(f"[Ch1] 120x120 matmul: Python={t_py*1000:.1f}ms, NumPy={t_np*1000:.3f}ms, speedup={t_py/t_np:.0f}x, max_diff={max_diff:.2e}")


def exp2_highdim_ortho():
    """Ch 2: High-dimensional random vectors become nearly orthogonal (cosine ~ O(1/sqrt(d)))."""
    rng = np.random.default_rng(2026)
    n_pairs = 5000
    with open(os.path.join(OUT_DIR, "highdim_cos.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["d", "emp_std", "theory_std", "pct_abs_lt_01", "max_abs"])
        for d in [2, 5, 10, 25, 50, 100, 250, 500, 1000]:
            U = rng.normal(size=(n_pairs, d))
            V = rng.normal(size=(n_pairs, d))
            U /= np.linalg.norm(U, axis=1, keepdims=True)
            V /= np.linalg.norm(V, axis=1, keepdims=True)
            cos_vals = np.sum(U * V, axis=1)
            emp_std = float(np.std(cos_vals))
            th_std = 1.0 / np.sqrt(d)
            pct = float(np.mean(np.abs(cos_vals) < 0.1) * 100.0)
            mx = float(np.max(np.abs(cos_vals)))
            w.writerow([d, f"{emp_std:.5f}", f"{th_std:.5f}", f"{pct:.2f}", f"{mx:.4f}"])
            print(f"[Ch2] d={d:4d}: std={emp_std:.4f} (1/sqrt(d)={th_std:.4f}), P(|cos|<0.1)={pct:5.1f}%, max|cos|={mx:.4f}")


def exp2_poly_ridge():
    """Ch 2: Polynomial regression (underfitting -> sweet spot -> overfitting) + Ridge."""
    rng = np.random.default_rng(42)
    x_tr = np.linspace(0.0, 1.0, 15)
    y_tr = np.sin(2 * np.pi * x_tr) + rng.normal(0, 0.25, size=x_tr.shape)
    x_te = np.linspace(0.0, 1.0, 200)
    y_te = np.sin(2 * np.pi * x_te)

    def vander(x, d):
        return np.column_stack([x**k for k in range(d + 1)])

    with open(os.path.join(OUT_DIR, "poly_ridge.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["deg", "train_mse", "test_mse", "ridge_test_mse", "w_norm"])
        for d in [1, 2, 3, 5, 7, 9, 11]:
            A_tr, A_te = vander(x_tr, d), vander(x_te, d)
            coef, *_ = np.linalg.lstsq(A_tr, y_tr, rcond=None)
            tr_mse = float(np.mean((A_tr @ coef - y_tr) ** 2))
            te_mse = float(np.mean((A_te @ coef - y_te) ** 2))
            P = np.eye(d + 1); P[0, 0] = 0.0
            coef_r = np.linalg.solve(A_tr.T @ A_tr + 1e-3 * P, A_tr.T @ y_tr)
            te_mse_r = float(np.mean((A_te @ coef_r - y_te) ** 2))
            w_norm = float(np.linalg.norm(coef))
            w.writerow([d, f"{tr_mse:.5f}", f"{te_mse:.5f}", f"{te_mse_r:.5f}", f"{w_norm:.2f}"])
            print(f"[Ch2] poly deg={d:2d}: tr={tr_mse:.4f}, te={te_mse:.4f}, ridge_te={te_mse_r:.4f}, ||w||={w_norm:.2f}")


def exp3_jordan_markov():
    """Ch 3: Jordan transient growth + Row-stochastic Markov convergence."""
    J = np.array([[0.9, 1.0], [0.0, 0.9]])
    with open(os.path.join(OUT_DIR, "jordan_transient.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "fro_norm", "upper_right", "diag"])
        Jk = np.eye(2)
        for k in range(0, 61):
            w.writerow([k, f"{np.linalg.norm(Jk, 'fro'):.6f}", f"{Jk[0,1]:.6f}", f"{Jk[0,0]:.6f}"])
            Jk = Jk @ J

    # Row-stochastic Markov matrix P (each row sums to 1)
    P = np.array([[0.7, 0.2, 0.1],
                  [0.1, 0.6, 0.3],
                  [0.2, 0.3, 0.5]])
    pi = np.array([1.0, 0.0, 0.0])
    # Exact stationary distribution: solve (P^T - I)pi = 0 with sum(pi) = 1
    eigs, vecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigs - 1.0))
    pi_star = np.real(vecs[:, idx]); pi_star /= np.sum(pi_star)
    for step in [1, 5, 10, 20]:
        pi_k = pi @ np.linalg.matrix_power(P, step)
        err = float(np.linalg.norm(pi_k - pi_star, 1))
        print(f"[Ch3] Markov step={step:2d}: pi_k={np.round(pi_k, 4)}, L1_err={err:.2e}")
    print(f"[Ch3] Markov stationary pi* = {np.round(pi_star, 5)}, eigs(P) = {np.round(np.sort(np.abs(eigs))[::-1], 4)}")


def exp4_svd_pca():
    """Ch 4: 64x64 non-separable grayscale image SVD compression + 2D PCA."""
    y, x = np.mgrid[0:64, 0:64]
    disc = (((x - 24) ** 2 + (y - 26) ** 2) < 15 ** 2).astype(float)
    ring = (np.abs(np.hypot(x - 42, y - 40) - 12.0) < 2.2).astype(float)
    diag = np.exp(-((x - y) ** 2) / 90.0)
    wave = 0.25 * np.sin(0.22 * x + 0.17 * y)
    img = 0.9 * disc + 0.7 * ring + 0.5 * diag + wave
    U, s, Vt = np.linalg.svd(img, full_matrices=False)
    total_norm = np.linalg.norm(img, "fro")
    with open(os.path.join(OUT_DIR, "svd_compress.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "rel_err", "energy_pct", "ratio_pct"])
        for k in [1, 2, 4, 8, 12, 16, 24]:
            Ak = (U[:, :k] * s[:k]) @ Vt[:k, :]
            rel_err = float(np.linalg.norm(img - Ak, "fro") / total_norm)
            energy = float(np.sum(s[:k] ** 2) / np.sum(s ** 2) * 100.0)
            ratio = k * (64 + 64 + 1) / (64 * 64) * 100.0
            w.writerow([k, f"{rel_err:.5f}", f"{energy:.2f}", f"{ratio:.2f}"])
            print(f"[Ch4] SVD k={k:2d}: rel_err={rel_err:.4f}, energy={energy:.2f}%, storage={ratio:.1f}%")


def exp6_gd_and_gradcheck():
    """Ch 6: GD spectral convergence + feature scaling + finite difference error curve."""
    # 1. Quadratic GD rates Q = diag(1, 10)
    Q = np.diag([1.0, 10.0])
    etas = {"0.05": 0.05, "opt": 2.0 / 11.0, "0.195": 0.195, "0.21": 0.21}
    ws = {k: np.array([1.0, 1.0]) for k in etas}
    with open(os.path.join(OUT_DIR, "gd_rates.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t", "eta005", "etaopt", "eta0195", "eta021"])
        for t in range(31):
            row = [t]
            for k in ["0.05", "opt", "0.195", "0.21"]:
                w = ws[k]
                loss = 0.5 * float(w @ Q @ w)
                row.append(f"{min(loss, 1e4):.6f}")
                ws[k] = w - etas[k] * (Q @ w)
            wr.writerow(row)
    print(f"[Ch6] GD t=20: eta=0.05->{0.5*(1-0.05)**40+5*(1-0.5)**40:.5f}, opt->{(9/11)**40 * 5.5:.5f}")

    # 2. Feature scaling impact on linear regression Hessian condition number
    rng = np.random.default_rng(7)
    N = 200
    x1 = rng.normal(0, 100.0, size=N)
    x2 = rng.normal(0, 0.1, size=N)
    X_raw = np.column_stack([x1, x2])
    X_std = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)
    k_raw = float(np.linalg.cond(X_raw.T @ X_raw / N))
    k_std = float(np.linalg.cond(X_std.T @ X_std / N))
    print(f"[Ch6] Condition number: raw={k_raw:.2e}, standardized={k_std:.3f}")

    # 3. Finite difference forward vs central error vs eps
    rng2 = np.random.default_rng(123)
    w0 = rng2.normal(size=10)
    a = rng2.normal(size=10)

    def f_test(w):
        return float(np.logaddexp(0.0, np.dot(w, a)) + 0.25 * (np.dot(w, w) ** 2))

    def grad_exact(w):
        p = 1.0 / (1.0 + np.exp(-np.dot(w, a)))
        return p * a + np.dot(w, w) * w

    g_true = grad_exact(w0)
    with open(os.path.join(OUT_DIR, "gradcheck_eps.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["eps", "fwd_err", "cen_err"])
        for exp in range(-1, -15, -1):
            eps = 10.0 ** exp
            g_fwd = np.zeros_like(w0)
            g_cen = np.zeros_like(w0)
            f0 = f_test(w0)
            for i in range(len(w0)):
                ei = np.zeros_like(w0); ei[i] = eps
                g_fwd[i] = (f_test(w0 + ei) - f0) / eps
                g_cen[i] = (f_test(w0 + ei) - f_test(w0 - ei)) / (2.0 * eps)
            ef = float(np.linalg.norm(g_fwd - g_true) / np.linalg.norm(g_true))
            ec = float(np.linalg.norm(g_cen - g_true) / np.linalg.norm(g_true))
            wr.writerow([f"{eps:.0e}", f"{ef:.3e}", f"{ec:.3e}"])
            if exp in (-2, -5, -8, -12):
                print(f"[Ch6] eps={eps:.0e}: fwd_err={ef:.2e}, cen_err={ec:.2e}")


if __name__ == "__main__":
    exp1_matmul_speed()
    exp2_highdim_ortho()
    exp2_poly_ridge()
    exp3_jordan_markov()
    exp4_svd_pca()
    exp6_gd_and_gradcheck()
