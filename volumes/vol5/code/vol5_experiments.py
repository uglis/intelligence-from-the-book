"""
Verified Python/NumPy/PyTorch experiments for AI Series Volume 5:
Deep Learning, Autodiff & Modern Architectures.
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


def exp1_micrograd_and_mlp():
    """Ch 1: From-scratch reverse-mode scalar/tensor autodiff verified against PyTorch + 2-moons MLP."""
    # Part A: Scalar reverse-mode autodiff verification vs PyTorch
    class Value:
        def __init__(self, data, _children=()):
            self.data = float(data)
            self.grad = 0.0
            self._backward = lambda: None
            self._prev = set(_children)

        def __add__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data + other.data, (self, other))
            def _b():
                self.grad += out.grad
                other.grad += out.grad
            out._backward = _b
            return out

        def __mul__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data * other.data, (self, other))
            def _b():
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
            out._backward = _b
            return out

        def tanh(self):
            t = math.tanh(self.data)
            out = Value(t, (self,))
            def _b():
                self.grad += (1.0 - t * t) * out.grad
            out._backward = _b
            return out

        def backward(self):
            topo, visited = [], set()
            def build(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        build(child)
                    topo.append(v)
            build(self)
            self.grad = 1.0
            for node in reversed(topo):
                node._backward()

    x1, x2, w1, w2, b = Value(2.0), Value(-1.5), Value(-0.8), Value(1.2), Value(0.5)
    y = (x1 * w1 + x2 * w2 + b).tanh()
    loss = (y + Value(-0.7)) * (y + Value(-0.7))
    loss.backward()

    tx1 = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    tx2 = torch.tensor(-1.5, dtype=torch.float64, requires_grad=True)
    tw1 = torch.tensor(-0.8, dtype=torch.float64, requires_grad=True)
    tw2 = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    ty = torch.tanh(tx1 * tw1 + tx2 * tw2 + tb)
    tloss = (ty - 0.7) ** 2
    tloss.backward()
    max_diff = max(
        abs(x1.grad - tx1.grad.item()),
        abs(w1.grad - tw1.grad.item()),
        abs(w2.grad - tw2.grad.item()),
        abs(b.grad - tb.grad.item()),
    )
    assert max_diff < 1e-12
    print(f"[Ch1] Micrograd vs PyTorch max_grad_diff={max_diff:.2e}, dL/dw1={w1.grad:.6f}, dL/dw2={w2.grad:.6f}")

    # Part B: Pure NumPy 2-layer ReLU MLP vs Linear Classifier on Two-Moons
    rng = np.random.default_rng(2026)
    n_half = 120
    t0 = rng.uniform(0, np.pi, n_half)
    X0 = np.column_stack([np.cos(t0), np.sin(t0)]) + rng.normal(0, 0.10, (n_half, 2))
    t1 = rng.uniform(0, np.pi, n_half)
    X1 = np.column_stack([1.0 - np.cos(t1), 0.5 - np.sin(t1)]) + rng.normal(0, 0.10, (n_half, 2))
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n_half), np.ones(n_half)])

    # Linear logistic baseline
    w_lin, b_lin = np.zeros(2), 0.0
    # 2-layer MLP (2 -> 32 -> 1)
    W1 = rng.normal(0, np.sqrt(2.0 / 2), (2, 32))
    b1 = np.zeros(32)
    W2 = rng.normal(0, np.sqrt(1.0 / 32), (32, 1))
    b2 = np.zeros(1)

    with open(os.path.join(OUT_DIR, "mlp_moons.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["epoch", "lin_loss", "mlp_loss", "lin_acc", "mlp_acc"])
        for ep in range(0, 601):
            # Linear forward
            zl = X @ w_lin + b_lin
            pl = 1.0 / (1.0 + np.exp(-np.clip(zl, -30, 30)))
            lin_loss = float(np.mean(np.logaddexp(0, zl) - y * zl))
            lin_acc = float(np.mean((pl >= 0.5) == y) * 100.0)
            # MLP forward
            h_pre = X @ W1 + b1
            h = np.maximum(0.0, h_pre)
            zm = (h @ W2 + b2).ravel()
            pm = 1.0 / (1.0 + np.exp(-np.clip(zm, -30, 30)))
            mlp_loss = float(np.mean(np.logaddexp(0, zm) - y * zm))
            mlp_acc = float(np.mean((pm >= 0.5) == y) * 100.0)

            if ep % 30 == 0:
                wr.writerow([ep, f"{lin_loss:.5f}", f"{mlp_loss:.5f}", f"{lin_acc:.2f}", f"{mlp_acc:.2f}"])
            if ep in (0, 200, 400, 600):
                print(f"[Ch1] ep={ep:3d}: Linear acc={lin_acc:.2f}% (loss={lin_loss:.4f}), MLP acc={mlp_acc:.2f}% (loss={mlp_loss:.4f})")

            # Linear backward
            dzl = (pl - y) / len(y)
            w_lin -= 0.4 * (X.T @ dzl)
            b_lin -= 0.4 * np.sum(dzl)
            # MLP manual VJP backward
            dzm = ((pm - y) / len(y))[:, None]
            dW2 = h.T @ dzm
            db2 = np.sum(dzm, axis=0)
            dh = dzm @ W2.T
            dh_pre = dh * (h_pre > 0)
            dW1 = X.T @ dh_pre
            db1 = np.sum(dh_pre, axis=0)
            lr = 0.45
            W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2


def exp2_init_norm_opt():
    """Ch 2: 20-layer ReLU activation std propagation (Small vs Large vs Kaiming vs LayerNorm)."""
    torch.manual_seed(2026)
    B, D, L = 64, 128, 20
    x0 = torch.randn(B, D)

    def forward_stds(mode):
        x = x0.clone()
        stds = [float(x.std().item())]
        for _ in range(L):
            if mode == "small":
                W = torch.randn(D, D) * (0.5 / math.sqrt(D))
            elif mode == "large":
                W = torch.randn(D, D) * (2.2 / math.sqrt(D))
            else:
                W = torch.randn(D, D) * math.sqrt(2.0 / D)  # Kaiming He
            x = F.relu(x @ W)
            if mode == "kaiming_ln":
                x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-5)
            stds.append(float(x.std().item()))
        return stds

    s_small = forward_stds("small")
    s_large = forward_stds("large")
    s_kaim = forward_stds("kaiming")
    s_ln = forward_stds("kaiming_ln")

    with open(os.path.join(OUT_DIR, "init_prop.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["layer", "small", "large", "kaiming", "kaiming_ln"])
        for l in range(L + 1):
            wr.writerow([l, f"{max(s_small[l], 1e-9):.4e}", f"{min(s_large[l], 1e6):.4e}", f"{s_kaim[l]:.4e}", f"{s_ln[l]:.4e}"])
    print(f"[Ch2] Layer 20 act std: Small={s_small[-1]:.2e}, Large={s_large[-1]:.2e}, Kaiming={s_kaim[-1]:.3f}, Kaiming+LN={s_ln[-1]:.3f}")


def exp3_resnet_grad_highway():
    """Ch 3: 24-layer Plain Net vs ResNet backward gradient norm across layers."""
    torch.manual_seed(42)
    B, D, L = 32, 64, 24

    def grad_norms(residual=False):
        x = torch.randn(B, D, requires_grad=True)
        h = x
        acts = []
        for _ in range(L):
            W = torch.randn(D, D) * (0.65 / math.sqrt(D))
            if residual:
                h = h + 0.35 * torch.tanh(h @ W)
            else:
                h = torch.tanh(h @ W)
            h.retain_grad()
            acts.append(h)
        loss = (acts[-1] ** 2).mean()
        loss.backward()
        return [float(a.grad.norm().item()) for a in acts]

    g_plain = grad_norms(residual=False)
    g_res = grad_norms(residual=True)

    with open(os.path.join(OUT_DIR, "resnet_grad.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["layer", "plain_grad", "resnet_grad"])
        for l in range(1, L + 1):
            wr.writerow([l, f"{max(g_plain[l-1], 1e-12):.4e}", f"{g_res[l-1]:.4e}"])
    print(f"[Ch3] Grad norm layer 1 vs 24: Plain {g_plain[0]:.2e} -> {g_plain[-1]:.2e} | ResNet {g_res[0]:.2e} -> {g_res[-1]:.2e}")


def exp4_mini_gpt_reverse():
    """Ch 4: Pure PyTorch 2-layer causal Mini-GPT trained on sequence reversal (e.g. a b c d | d c b a)."""
    torch.manual_seed(2026)
    vocab_size = 12  # tokens 0..9 + sep(10) + pad(11)
    seq_half = 6
    seq_len = 2 * seq_half + 1  # 6 + sep + 6 = 13

    def make_batch(bs):
        src = torch.randint(0, 10, (bs, seq_half))
        sep = torch.full((bs, 1), 10, dtype=torch.long)
        tgt = torch.flip(src, dims=[1])
        seq = torch.cat([src, sep, tgt], dim=1)
        return seq[:, :-1], seq[:, 1:]  # input len 8, target len 8

    class CausalSelfAttn(nn.Module):
        def __init__(self, d_model, n_head, max_len, scale=True):
            super().__init__()
            self.n_head = n_head
            self.d_head = d_model // n_head
            self.scale = scale
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.proj = nn.Linear(d_model, d_model, bias=False)
            self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))

        def forward(self, x):
            B, T, C = x.shape
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
            att = q @ k.transpose(-2, -1)
            if self.scale:
                att = att / math.sqrt(self.d_head)
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
            return self.proj(y)

    class MiniGPT(nn.Module):
        def __init__(self, d_model=48, n_head=4, n_layer=2):
            super().__init__()
            self.tok = nn.Embedding(vocab_size, d_model)
            self.pos = nn.Embedding(seq_len, d_model)
            self.blocks = nn.ModuleList([
                nn.ModuleDict({
                    "ln1": nn.LayerNorm(d_model),
                    "attn": CausalSelfAttn(d_model, n_head, seq_len),
                    "ln2": nn.LayerNorm(d_model),
                    "ffn": nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
                }) for _ in range(n_layer)
            ])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx):
            B, T = idx.shape
            x = self.tok(idx) + self.pos(torch.arange(T))
            for blk in self.blocks:
                x = x + blk["attn"](blk["ln1"](x))
                x = x + blk["ffn"](blk["ln2"](x))
            return self.head(self.ln_f(x))

    model = MiniGPT()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    X_val, Y_val = make_batch(256)

    @torch.no_grad()
    def eval_autoregressive(m, src_with_sep, tgt):
        """True autoregressive greedy decoding: feed ONLY [src, sep] and generate token by token."""
        cur = src_with_sep.clone()
        for _ in range(tgt.shape[1]):
            next_tok = m(cur)[:, -1, :].argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, next_tok], dim=1)
        gen = cur[:, -tgt.shape[1]:]
        return float((gen == tgt).all(dim=-1).float().mean().item() * 100.0)

    with open(os.path.join(OUT_DIR, "transformer_train.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["step", "loss", "val_tok_acc", "val_seq_acc"])
        for step in range(0, 321):
            xb, yb = make_batch(64)
            logits = model(xb)
            # Compute loss ONLY on the reversed answer half (positions seq_half..)
            loss = F.cross_entropy(logits[:, seq_half:, :].reshape(-1, vocab_size), yb[:, seq_half:].reshape(-1))
            if step % 20 == 0:
                with torch.no_grad():
                    v_logits = model(X_val)[:, seq_half:, :]
                    v_pred = v_logits.argmax(dim=-1)
                    v_tgt = Y_val[:, seq_half:]
                    tok_acc = float((v_pred == v_tgt).float().mean().item() * 100.0)
                    # True autoregressive greedy decoding exact-match sequence accuracy!
                    seq_acc = eval_autoregressive(model, X_val[:, :seq_half + 1], v_tgt)
                wr.writerow([step, f"{loss.item():.5f}", f"{tok_acc:.2f}", f"{seq_acc:.2f}"])
                if step in (0, 20, 40, 60, 80, 160, 240, 320):
                    print(f"[Ch4] Mini-GPT step={step:3d}: loss={loss.item():.4f}, tf_tok_acc={tok_acc:.2f}%, ar_seq_acc={seq_acc:.2f}%")
            opt.zero_grad()
            loss.backward()
            opt.step()


def exp5_infonce_contrastive():
    """Ch 5: CLIP-style symmetric InfoNCE contrastive alignment between two modalities."""
    torch.manual_seed(2026)
    N, D_in, D_emb = 128, 16, 16
    # Shared latent concepts Z mapped through two distinct random nonlinear modalities A & B
    Z = torch.randn(N, 8)
    Wa = torch.randn(8, D_in)
    Wb = torch.randn(8, D_in)
    Xa = torch.tanh(Z @ Wa) + 0.08 * torch.randn(N, D_in)
    Xb = torch.sin(Z @ Wb) + 0.08 * torch.randn(N, D_in)

    enc_a = nn.Sequential(nn.Linear(D_in, 32), nn.ReLU(), nn.Linear(32, D_emb))
    enc_b = nn.Sequential(nn.Linear(D_in, 32), nn.ReLU(), nn.Linear(32, D_emb))
    opt = torch.optim.AdamW(list(enc_a.parameters()) + list(enc_b.parameters()), lr=5e-3)
    tau = 0.15
    labels = torch.arange(N)

    with open(os.path.join(OUT_DIR, "infonce_train.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["epoch", "loss", "top1_acc", "pos_cos", "neg_cos"])
        for ep in range(0, 201):
            za = F.normalize(enc_a(Xa), dim=-1)
            zb = F.normalize(enc_b(Xb), dim=-1)
            sim = za @ zb.T  # (N, N)
            logits = sim / tau
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
            if ep % 10 == 0:
                with torch.no_grad():
                    top1 = float((sim.argmax(dim=-1) == labels).float().mean().item() * 100.0)
                    pos_c = float(torch.diag(sim).mean().item())
                    neg_c = float((sim.sum() - torch.diag(sim).sum()).item() / (N * (N - 1)))
                wr.writerow([ep, f"{loss.item():.5f}", f"{top1:.2f}", f"{pos_c:.4f}", f"{neg_c:.4f}"])
                if ep in (0, 50, 100, 200):
                    print(f"[Ch5] InfoNCE ep={ep:3d}: loss={loss.item():.4f}, top1={top1:.2f}%, pos_cos={pos_c:.3f}, neg_cos={neg_c:.3f}")
            opt.zero_grad()
            loss.backward()
            opt.step()


def exp6_ddpm_2d_ring():
    """Ch 6: Pure PyTorch 2D DDPM Diffusion Model on a 2D ring/cluster mixture."""
    torch.manual_seed(2026)
    T = 40
    betas = torch.linspace(1e-3, 0.08, T)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)

    # Target 2D distribution: 4-mode Gaussian ring of radius 2.0
    centers = torch.tensor([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]])
    def sample_data(n):
        idx = torch.randint(0, 4, (n,))
        return centers[idx] + 0.18 * torch.randn(n, 2)

    eps_net = nn.Sequential(
        nn.Linear(3, 64), nn.SiLU(),
        nn.Linear(64, 64), nn.SiLU(),
        nn.Linear(64, 2)
    )
    opt = torch.optim.AdamW(eps_net.parameters(), lr=4e-3)

    @torch.no_grad()
    def sample_ddpm(n=400):
        x = torch.randn(n, 2)
        for t in reversed(range(T)):
            t_in = torch.full((n, 1), (t + 1) / T)
            eps_pred = eps_net(torch.cat([x, t_in], dim=-1))
            c1 = 1.0 / math.sqrt(alphas[t].item())
            c2 = (1.0 - alphas[t].item()) / math.sqrt(1.0 - abar[t].item())
            mean = c1 * (x - c2 * eps_pred)
            if t > 0:
                x = mean + math.sqrt(betas[t].item()) * torch.randn_like(x)
            else:
                x = mean
        return x

    def eval_ring_error(samples):
        # Mean distance of each generated point to nearest of the 4 true centers
        dists = torch.cdist(samples, centers)
        return float(dists.min(dim=-1).values.mean().item())

    real_err = eval_ring_error(sample_data(400))

    with open(os.path.join(OUT_DIR, "ddpm_train.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["step", "eps_mse", "gen_err", "real_err"])
        for step in range(0, 601):
            x0 = sample_data(256)
            t = torch.randint(0, T, (256,))
            eps = torch.randn_like(x0)
            ab = abar[t].unsqueeze(-1)
            xt = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * eps
            t_in = ((t.float() + 1.0) / T).unsqueeze(-1)
            eps_hat = eps_net(torch.cat([xt, t_in], dim=-1))
            loss = F.mse_loss(eps_hat, eps)
            if step % 60 == 0:
                gen_err = eval_ring_error(sample_ddpm(400))
                wr.writerow([step, f"{loss.item():.5f}", f"{gen_err:.4f}", f"{real_err:.4f}"])
                if step in (0, 180, 360, 600):
                    print(f"[Ch6] DDPM step={step:3d}: eps_mse={loss.item():.4f}, gen_dist_to_mode={gen_err:.3f} (real={real_err:.3f})")
            opt.zero_grad()
            loss.backward()
            opt.step()


if __name__ == "__main__":
    exp1_micrograd_and_mlp()
    exp2_init_norm_opt()
    exp3_resnet_grad_highway()
    exp4_mini_gpt_reverse()
    exp5_infonce_contrastive()
    exp6_ddpm_2d_ring()
