"""
Verified Python/NumPy experiments for AI Series Volume 3:
Classical AI — Search (A*), Games (Alpha-Beta & MCTS), CSP (AC-3 & Min-Conflicts),
Bayesian Networks (Variable Elimination & Sampling), HMM (Viterbi), and MDP (VI & PI).
Generates real CSV files in ../data/ for pgfplots and prints verified metrics.
"""
import os
import csv
import math
import time
import heapq
from collections import deque
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def exp1_maze_astar():
    """Ch 1: 25x25 obstacle grid comparing BFS, UCS, A*(Euclid), A*(Manhattan), Greedy."""
    H, W = 25, 25
    grid = np.zeros((H, W), dtype=int)
    # U-shaped trap + central walls so Greedy takes a suboptimal detour while A* beats UCS dramatically
    grid[6:20, 12] = 1
    grid[6, 6:13] = 1
    grid[19, 6:13] = 1
    grid[2:14, 18] = 1
    start, goal = (12, 4), (12, 22)
    grid[start] = 0; grid[goal] = 0

    def neighbors(r, c):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0:
                yield (nr, nc), 1.0

    def search(mode):
        # Standard A* tie-breaking: (f, h, node) -> prefer smaller h (larger g) on ties
        pq = [(0.0, 0.0, 0.0, start)]
        g_best = {start: 0.0}
        expanded = 0
        closed = set()
        while pq:
            f, h_tie, g, u = heapq.heappop(pq)
            if u in closed:
                continue
            closed.add(u)
            expanded += 1
            if u == goal:
                return g, expanded
            for v, cost in neighbors(*u):
                ng = g + cost
                if ng < g_best.get(v, float("inf")):
                    g_best[v] = ng
                    if mode == "ucs":
                        h = 0.0
                        key = (ng, 0.0)
                    elif mode == "euclid":
                        h = math.hypot(v[0] - goal[0], v[1] - goal[1])
                        key = (ng + h, h)
                    elif mode == "manhattan":
                        h = abs(v[0] - goal[0]) + abs(v[1] - goal[1])
                        key = (ng + h, h)
                    elif mode == "greedy":
                        h = abs(v[0] - goal[0]) + abs(v[1] - goal[1])
                        key = (h, ng)
                    heapq.heappush(pq, (key[0], key[1], ng, v))
        return float("inf"), expanded

    q = deque([(start, 0)])
    vis = {start}
    bfs_exp = 0
    bfs_cost = -1
    while q:
        u, d = q.popleft()
        bfs_exp += 1
        if u == goal:
            bfs_cost = d
            break
        for v, _ in neighbors(*u):
            if v not in vis:
                vis.add(v)
                q.append((v, d + 1))

    print(f"[Ch1] 25x25 Maze: BFS        cost={bfs_cost}, expanded={bfs_exp}")
    for mode in ["ucs", "euclid", "manhattan", "greedy"]:
        cost, exp = search(mode)
        print(f"[Ch1] 25x25 Maze: {mode:10s} cost={cost:.0f}, expanded={exp}")


def exp2_tictactoe_alphabeta_mcts():
    """Ch 2: Tic-Tac-Toe Minimax vs Alpha-Beta node counts + MCTS win-rate curve."""
    WINS = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

    def winner(b):
        for i, j, k in WINS:
            if b[i] != 0 and b[i] == b[j] == b[k]:
                return b[i]
        return 0

    nodes_mm = 0
    def minimax(b, player):
        nonlocal nodes_mm
        nodes_mm += 1
        w = winner(b)
        if w != 0:
            return w
        moves = [i for i in range(9) if b[i] == 0]
        if not moves:
            return 0
        if player == 1:
            best = -2
            for m in moves:
                b[m] = 1
                best = max(best, minimax(b, -1))
                b[m] = 0
            return best
        else:
            best = 2
            for m in moves:
                b[m] = -1
                best = min(best, minimax(b, 1))
                b[m] = 0
            return best

    nodes_ab = 0
    def alphabeta(b, player, alpha, beta, ordered=False):
        nonlocal nodes_ab
        nodes_ab += 1
        w = winner(b)
        if w != 0:
            return w
        order = [4, 0, 2, 6, 8, 1, 3, 5, 7] if ordered else list(range(9))
        moves = [i for i in order if b[i] == 0]
        if not moves:
            return 0
        if player == 1:
            val = -2
            for m in moves:
                b[m] = 1
                val = max(val, alphabeta(b, -1, alpha, beta, ordered))
                b[m] = 0
                alpha = max(alpha, val)
                if alpha >= beta:
                    break
            return val
        else:
            val = 2
            for m in moves:
                b[m] = -1
                val = min(val, alphabeta(b, 1, alpha, beta, ordered))
                b[m] = 0
                beta = min(beta, val)
                if alpha >= beta:
                    break
            return val

    v_mm = minimax([0]*9, 1)
    v_ab_nat = alphabeta([0]*9, 1, -2, 2, ordered=False)
    n_ab_nat = nodes_ab
    nodes_ab = 0
    v_ab_ord = alphabeta([0]*9, 1, -2, 2, ordered=True)
    n_ab_ord = nodes_ab
    print(f"[Ch2] Tic-Tac-Toe root val={v_mm}: Minimax={nodes_mm:,} nodes, AB(natural)={n_ab_nat:,} ({100*(1-n_ab_nat/nodes_mm):.1f}% pruned), AB(ordered)={n_ab_ord:,} ({100*(1-n_ab_ord/nodes_mm):.1f}% pruned)")

    # MCTS vs random opponent on Tic-Tac-Toe
    rng = np.random.default_rng(2026)

    class Node:
        __slots__ = ("state", "player", "move", "parent", "children", "untried", "wins", "visits")
        def __init__(self, state, player, move=None, parent=None):
            self.state = state
            self.player = player
            self.move = move
            self.parent = parent
            self.children = []
            self.untried = [i for i in range(9) if state[i] == 0] if winner(state) == 0 else []
            self.wins = 0.0
            self.visits = 0

    def mcts_move(board, player, iters):
        root = Node(board[:], player)
        for _ in range(iters):
            node = root
            # 1. Selection
            while not node.untried and node.children:
                log_n = math.log(node.visits)
                node = max(node.children, key=lambda ch: (ch.wins / ch.visits) + 1.414 * math.sqrt(log_n / ch.visits))
            # 2. Expansion
            if node.untried:
                idx = int(rng.integers(0, len(node.untried)))
                m = node.untried.pop(idx)
                ns = node.state[:]
                ns[m] = node.player
                child = Node(ns, -node.player, move=m, parent=node)
                node.children.append(child)
                node = child
            # 3. Rollout
            sim = node.state[:]
            cur = node.player
            while True:
                w = winner(sim)
                if w != 0:
                    break
                avail = [i for i in range(9) if sim[i] == 0]
                if not avail:
                    w = 0
                    break
                sim[avail[int(rng.integers(0, len(avail)))]] = cur
                cur = -cur
            # 4. Backprop (reward from parent's perspective who moved to child)
            while node is not None:
                node.visits += 1
                mover = -node.player
                if w == mover:
                    node.wins += 1.0
                elif w == 0:
                    node.wins += 0.5
                node = node.parent
        best = max(root.children, key=lambda ch: ch.visits)
        return best.move

    with open(os.path.join(OUT_DIR, "mcts_winrate.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["iters", "win_pct", "nonloss_pct"])
        for iters in [5, 10, 25, 50, 100, 200, 400]:
            wins = draws = 0
            games = 120
            for _ in range(games):
                b = [0]*9
                cur = 1
                while True:
                    w = winner(b)
                    avail = [i for i in range(9) if b[i] == 0]
                    if w != 0 or not avail:
                        break
                    if cur == 1:
                        m = mcts_move(b, 1, iters)
                    else:
                        m = avail[int(rng.integers(0, len(avail)))]
                    b[m] = cur
                    cur = -cur
                if w == 1:
                    wins += 1
                elif w == 0:
                    draws += 1
            wr.writerow([iters, f"{100*wins/games:.2f}", f"{100*(wins+draws)/games:.2f}"])
            print(f"[Ch2] MCTS iters={iters:3d}: win={100*wins/games:5.1f}%, win+draw={100*(wins+draws)/games:5.1f}%")


def exp3_nqueens_csp():
    """Ch 3: N-Queens comparing Naive Backtracking vs MRV+FC vs Min-Conflicts."""
    def solve_naive(n):
        nodes = 0
        cols = [-1] * n
        def bt(r):
            nonlocal nodes
            if r == n:
                return True
            for c in range(n):
                nodes += 1
                ok = True
                for pr in range(r):
                    if cols[pr] == c or abs(cols[pr] - c) == r - pr:
                        ok = False; break
                if ok:
                    cols[r] = c
                    if bt(r + 1):
                        return True
                    cols[r] = -1
            return False
        t0 = time.perf_counter()
        bt(0)
        return nodes, (time.perf_counter() - t0) * 1000.0

    def solve_mrv_fc(n):
        nodes = 0
        domains = [set(range(n)) for _ in range(n)]
        assign = {}
        def bt():
            nonlocal nodes
            if len(assign) == n:
                return True
            # MRV
            var = min((v for v in range(n) if v not in assign), key=lambda v: len(domains[v]))
            for val in sorted(domains[var]):
                nodes += 1
                pruned = []
                wipeout = False
                for other in range(n):
                    if other not in assign and other != var:
                        bad = {val, val + (other - var), val - (other - var)} & domains[other]
                        if bad:
                            domains[other] -= bad
                            pruned.append((other, bad))
                            if not domains[other]:
                                wipeout = True; break
                if not wipeout:
                    assign[var] = val
                    if bt():
                        return True
                    del assign[var]
                for other, bad in pruned:
                    domains[other] |= bad
            return False
        t0 = time.perf_counter()
        bt()
        return nodes, (time.perf_counter() - t0) * 1000.0

    def solve_min_conflicts(n, max_steps=10000):
        rng = np.random.default_rng(42)
        t0 = time.perf_counter()
        q = rng.integers(0, n, size=n)
        for step in range(1, max_steps + 1):
            conf_rows = []
            for r in range(n):
                for r2 in range(n):
                    if r2 != r and (q[r2] == q[r] or abs(q[r2] - q[r]) == abs(r2 - r)):
                        conf_rows.append(r); break
            if not conf_rows:
                return step, (time.perf_counter() - t0) * 1000.0
            r = conf_rows[int(rng.integers(0, len(conf_rows)))]
            best_c, best_val = [], n + 1
            for c in range(n):
                cnt = sum(1 for r2 in range(n) if r2 != r and (q[r2] == c or abs(q[r2] - c) == abs(r2 - r)))
                if cnt < best_val:
                    best_val = cnt; best_c = [c]
                elif cnt == best_val:
                    best_c.append(c)
            q[r] = best_c[int(rng.integers(0, len(best_c)))]
        return max_steps, (time.perf_counter() - t0) * 1000.0

    for n in [8, 12, 16, 20]:
        nn, tn = solve_naive(n) if n <= 16 else (-1, -1.0)
        nm, tm = solve_mrv_fc(n)
        ns, ts = solve_min_conflicts(n)
        print(f"[Ch3] N={n:2d}: Naive={nn} ({tn:.2f}ms) | MRV+FC={nm} ({tm:.2f}ms) | MinConflicts={ns} steps ({ts:.2f}ms)")
    ns50, ts50 = solve_min_conflicts(50)
    print(f"[Ch3] N=50: MinConflicts={ns50} steps ({ts50:.2f}ms)")


def exp4_exp5_alarm_bn_and_hmm():
    """Ch 4 & 5: Pearl's Alarm BN exact VE + Likelihood Weighting vs Gibbs convergence + Weather-Umbrella HMM."""
    # Alarm BN: B, E -> A -> J, M
    pB = {1: 0.001, 0: 0.999}
    pE = {1: 0.002, 0: 0.998}
    pA = {(1,1): 0.95, (1,0): 0.94, (0,1): 0.29, (0,0): 0.001}
    pJ = {1: 0.90, 0: 0.05}
    pM = {1: 0.70, 0: 0.01}

    # Exact P(B=1 | J=1, M=1)
    num, den = 0.0, 0.0
    for b in (0, 1):
        for e in (0, 1):
            for a in (0, 1):
                pa = pA[(b, e)] if a == 1 else (1 - pA[(b, e)])
                prob = pB[b] * pE[e] * pa * pJ[a] * pM[a]
                den += prob
                if b == 1:
                    num += prob
    exact_b_jm = num / den
    print(f"[Ch4] Alarm BN exact: P(J=1,M=1)={den:.7f}, P(B=1 | J=1, M=1)={exact_b_jm:.6f}")

    # Ch 5: Likelihood Weighting vs Gibbs Sampling error vs M
    rng = np.random.default_rng(2026)
    with open(os.path.join(OUT_DIR, "bn_sampling.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["M", "lw_err", "gibbs_err"])
        for M in [100, 300, 1000, 3000, 10000, 30000]:
            # Likelihood Weighting
            b_s = (rng.random(M) < 0.001).astype(int)
            e_s = (rng.random(M) < 0.002).astype(int)
            pa_s = np.where((b_s==1)&(e_s==1), 0.95, np.where((b_s==1)&(e_s==0), 0.94, np.where((b_s==0)&(e_s==1), 0.29, 0.001)))
            a_s = (rng.random(M) < pa_s).astype(int)
            w_s = np.where(a_s==1, 0.90*0.70, 0.05*0.01)
            est_lw = float(np.sum(w_s[b_s==1]) / np.sum(w_s))

            # Gibbs Sampling (burn-in 200)
            b_g, e_g, a_g = 0, 0, 1
            cnt_b1 = 0
            burn = 200
            for t in range(M + burn):
                # sample B | E, A
                pa1 = pA[(1, e_g)] if a_g else (1 - pA[(1, e_g)])
                pa0 = pA[(0, e_g)] if a_g else (1 - pA[(0, e_g)])
                pb1 = (0.001 * pa1) / (0.001 * pa1 + 0.999 * pa0)
                b_g = 1 if rng.random() < pb1 else 0
                # sample E | B, A
                pe1_a = pA[(b_g, 1)] if a_g else (1 - pA[(b_g, 1)])
                pe0_a = pA[(b_g, 0)] if a_g else (1 - pA[(b_g, 0)])
                pe1 = (0.002 * pe1_a) / (0.002 * pe1_a + 0.998 * pe0_a)
                e_g = 1 if rng.random() < pe1 else 0
                # sample A | B, E, J=1, M=1
                pa_be = pA[(b_g, e_g)]
                p_a1 = (pa_be * 0.90 * 0.70) / (pa_be * 0.90 * 0.70 + (1 - pa_be) * 0.05 * 0.01)
                a_g = 1 if rng.random() < p_a1 else 0
                if t >= burn and b_g == 1:
                    cnt_b1 += 1
            est_gibbs = cnt_b1 / M
            wr.writerow([M, f"{abs(est_lw - exact_b_jm):.5f}", f"{abs(est_gibbs - exact_b_jm):.5f}"])
            if M in (1000, 10000, 30000):
                print(f"[Ch5] BN M={M:5d}: LW={est_lw:.4f} (err={abs(est_lw-exact_b_jm):.4f}), Gibbs={est_gibbs:.4f} (err={abs(est_gibbs-exact_b_jm):.4f})")

    # Ch 5: Weather-Umbrella HMM (Rain=1, Sun=0), obs=[U, U, ~U, U, U]
    T_mat = np.array([[0.7, 0.3], [0.3, 0.7]])  # [Rain, Sun] -> [Rain, Sun]
    E_mat = np.array([[0.9, 0.1], [0.2, 0.8]])  # state x [Umbrella=0, NoUmb=1]
    obs = [0, 0, 1, 0, 0]
    pi0 = np.array([0.5, 0.5])

    # Forward filtering
    f_list = []
    f_cur = pi0
    for o in obs:
        pred = f_cur @ T_mat
        f_cur = pred * E_mat[:, o]
        f_cur /= f_cur.sum()
        f_list.append(f_cur.copy())
    # Backward smoothing
    b_list = [np.ones(2)] * len(obs)
    for t in range(len(obs) - 2, -1, -1):
        b_list[t] = T_mat @ (E_mat[:, obs[t + 1]] * b_list[t + 1])
        b_list[t] /= b_list[t].sum()
    smooth = [(f_list[t] * b_list[t]) / np.sum(f_list[t] * b_list[t]) for t in range(len(obs))]

    # Viterbi (log-space)
    dp = np.zeros((len(obs), 2))
    ptr = np.zeros((len(obs), 2), dtype=int)
    dp[0] = np.log(pi0 @ T_mat) + np.log(E_mat[:, obs[0]])
    for t in range(1, len(obs)):
        for s in range(2):
            scores = dp[t - 1] + np.log(T_mat[:, s]) + np.log(E_mat[s, obs[t]])
            ptr[t, s] = int(np.argmax(scores))
            dp[t, s] = scores[ptr[t, s]]
    path = [int(np.argmax(dp[-1]))]
    for t in range(len(obs) - 1, 0, -1):
        path.append(int(ptr[t, path[-1]]))
    path.reverse()
    print("[Ch5] HMM Rain probs (Filter):", [round(float(x[0]), 4) for x in f_list])
    print("[Ch5] HMM Rain probs (Smooth):", [round(float(x[0]), 4) for x in smooth])
    print("[Ch5] HMM Viterbi path (0=Rain,1=Sun):", path)


def exp6_mdp_gridworld():
    """Ch 6: Classic 4x3 stochastic Gridworld (Russell-Norvig) — Value Iteration vs Policy Iteration."""
    # Grid: rows 0..2 (0=bottom, 2=top), cols 0..3; wall at (1,1); terminals: (2,3)=+1, (1,3)=-1
    states = [(r, c) for r in range(3) for c in range(4) if (r, c) != (1, 1)]
    terminals = {(2, 3): 1.0, (1, 3): -1.0}
    actions = ["N", "S", "E", "W"]
    dirs = {"N": (1, 0), "S": (-1, 0), "E": (0, 1), "W": (0, -1)}
    left_right = {"N": ("W", "E"), "S": ("E", "W"), "E": ("N", "S"), "W": ("S", "N")}
    gamma = 0.9
    step_r = -0.04

    def move(s, d):
        r, c = s
        dr, dc = dirs[d]
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 4 and (nr, nc) != (1, 1):
            return (nr, nc)
        return s

    def transitions(s, a):
        if s in terminals:
            return []
        l, r = left_right[a]
        res = {}
        for d, p in [(a, 0.8), (l, 0.1), (r, 0.1)]:
            ns = move(s, d)
            res[ns] = res.get(ns, 0.0) + p
        return list(res.items())

    # Exact optimal V* via Policy Iteration
    pi = {s: "N" for s in states if s not in terminals}
    pi_iters = 0
    while True:
        pi_iters += 1
        # Solve linear system (I - gamma P^pi) V = R^pi
        idx = {s: i for i, s in enumerate(states)}
        n = len(states)
        A = np.eye(n); b = np.zeros(n)
        for s in states:
            i = idx[s]
            if s in terminals:
                b[i] = terminals[s]
            else:
                b[i] = step_r
                for ns, p in transitions(s, pi[s]):
                    A[i, idx[ns]] -= gamma * p
        V_pi = np.linalg.solve(A, b)
        V_dict = {s: float(V_pi[idx[s]]) for s in states}
        stable = True
        for s in states:
            if s in terminals:
                continue
            best_a = max(actions, key=lambda a: step_r + gamma * sum(p * V_dict[ns] for ns, p in transitions(s, a)))
            if best_a != pi[s]:
                pi[s] = best_a
                stable = False
        if stable:
            break
    V_star = V_dict

    # Value Iteration convergence curve
    V = {s: (terminals[s] if s in terminals else 0.0) for s in states}
    with open(os.path.join(OUT_DIR, "mdp_convergence.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["iter", "inf_err", "gamma_bound"])
        init_err = max(abs(V[s] - V_star[s]) for s in states)
        wr.writerow([0, f"{init_err:.6f}", f"{init_err:.6f}"])
        for it in range(1, 36):
            V_new = {}
            for s in states:
                if s in terminals:
                    V_new[s] = terminals[s]
                else:
                    V_new[s] = max(step_r + gamma * sum(p * V[ns] for ns, p in transitions(s, a)) for a in actions)
            V = V_new
            err = max(abs(V[s] - V_star[s]) for s in states)
            wr.writerow([it, f"{max(err, 1e-6):.6f}", f"{init_err * (gamma ** it):.6f}"])

    print(f"[Ch6] Gridworld Policy Iteration converged in {pi_iters} iterations!")
    print("[Ch6] Optimal V* (rows top=2 to bottom=0):")
    for r in [2, 1, 0]:
        row_s = []
        for c in range(4):
            if (r, c) == (1, 1):
                row_s.append("  WALL ")
            else:
                row_s.append(f"{V_star[(r,c)]:+6.3f}")
        print("  " + " ".join(row_s))
    print("[Ch6] Optimal Policy pi*:")
    for r in [2, 1, 0]:
        row_p = []
        for c in range(4):
            if (r, c) == (1, 1):
                row_p.append("WALL")
            elif (r, c) in terminals:
                row_p.append(f"{terminals[(r,c)]:+4.0f}")
            else:
                row_p.append(f"  {pi[(r,c)]} ")
        print("  " + " ".join(row_p))


if __name__ == "__main__":
    exp1_maze_astar()
    exp2_tictactoe_alphabeta_mcts()
    exp3_nqueens_csp()
    exp4_exp5_alarm_bn_and_hmm()
    exp6_mdp_gridworld()
