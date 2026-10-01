"""
Author: Yang Shao
Institution: University of Texas at Austin
Description: Numerical experiments for the detectability lemma (DL) and for a
             discretized frustration-free adiabatic path.

  Experiment 1 (DL contraction): repeatedly apply the DL operator of a fixed
      random frustration-free Hamiltonian to |+>^n and track the overlap with
      its ground state.

  Experiment 2 (adiabatic path): follow a path H(s), s in [0, 1], of
      frustration-free Hamiltonians from one whose ground state is |+>^n to a
      random target. At each of T steps, apply one DL pass of H(s_t) using
      postselected projective measurements, and record the probability that
      every measurement accepts. Compare with postselecting directly onto the
      target ground state (no path).

Earlier versions consulted ChatGPT for debugging, formatting and plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, eigsh

OUT = "."  # output directory for figures


# ---------------------------------------------------------------------------
# State-vector helpers
# ---------------------------------------------------------------------------

def apply_local(op, qubits, psi, n):
    """Apply a 2^k x 2^k operator on `qubits` to an n-qubit state vector."""
    k = len(qubits)
    t = psi.reshape((2,) * n)
    t = np.tensordot(op.reshape((2,) * (2 * k)), t,
                     axes=(list(range(k, 2 * k)), qubits))
    t = np.moveaxis(t, list(range(k)), qubits)
    return t.reshape(-1)


def product_state(qubit_states):
    psi = qubit_states[0]
    for q in qubit_states[1:]:
        psi = np.kron(psi, q)
    return psi


def random_qubit(rng):
    v = rng.random(2)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Random frustration-free Hamiltonians with product ground states
# ---------------------------------------------------------------------------

def random_supports(n, m, k, rng):
    """m random k-subsets of n qubits, covering every qubit at least once."""
    while True:
        supports = [sorted(rng.choice(n, size=k, replace=False).tolist())
                    for _ in range(m)]
        if len(set(q for s in supports for q in s)) == n:
            return supports


def penalized_projector(local_gs, w):
    """Projector onto the span of the columns of w (a 2^k x r matrix) after
    removing their component along the local ground state."""
    w = w - np.outer(local_gs, local_gs.conj() @ w)
    q, _ = np.linalg.qr(w)
    return q @ q.conj().T


def hamiltonian_terms(qubit_states, supports, raw_vectors):
    """Rank-r projector terms whose kernel contains the restriction of the
    product state to the support of each term."""
    terms = []
    for supp, w in zip(supports, raw_vectors):
        local_gs = product_state([qubit_states[q] for q in supp])
        terms.append((penalized_projector(local_gs, w), supp))
    return terms


def layers_of(supports):
    """Greedy partition of term indices into layers of disjoint supports."""
    layers = []
    for a, supp in enumerate(supports):
        for layer in layers:
            if all(not set(supp) & set(supports[b]) for b in layer):
                layer.append(a)
                break
        else:
            layers.append([a])
    return layers


def spectral_gap(terms, n):
    """Gap above the zero ground energy (frustration-free, so E0 = 0)."""
    dim = 2 ** n

    def matvec(x):
        x = np.asarray(x).reshape(-1)
        out = np.zeros(dim, dtype=complex)
        for h, supp in terms:
            out += apply_local(h, supp, x, n)
        return out

    op = LinearOperator((dim, dim), matvec=matvec, dtype=complex)
    vals = np.sort(eigsh(op, k=2, which="SA", tol=1e-10,
                         return_eigenvectors=False).real)
    return vals[0], vals[1] - vals[0]


class Path:
    """H(s) = sum_a Pi_a(s), with product ground state |g(s)> = (x) |g_j(s)>.
    Each qubit state interpolates from |+> to a random target. Each term
    Pi_a(s) is a rank-r projector on k qubits whose range interpolates
    between two random r-dimensional subspaces, with the local ground state
    removed. H(s) is
    frustration-free for every s, and its spectrum genuinely varies with s."""

    def __init__(self, n, m, k, rng, rank=4):
        self.n, self.k = n, k
        self.supports = random_supports(n, m, k, rng)
        self.layers = layers_of(self.supports)
        self.q_in = [np.ones(2) / np.sqrt(2) for _ in range(n)]
        self.q_f = [random_qubit(rng) for _ in range(n)]
        cplx = lambda: (rng.standard_normal((2 ** k, rank))
                        + 1j * rng.standard_normal((2 ** k, rank)))
        self.w_in = [cplx() for _ in range(m)]
        self.w_f = [cplx() for _ in range(m)]

    def qubits(self, s):
        out = []
        for a, b in zip(self.q_in, self.q_f):
            v = (1 - s) * a + s * b
            out.append(v / np.linalg.norm(v))
        return out

    def terms(self, s):
        raw = [(1 - s) * a + s * b for a, b in zip(self.w_in, self.w_f)]
        return hamiltonian_terms(self.qubits(s), self.supports, raw)

    def ground_state(self, s):
        return product_state(self.qubits(s))


# ---------------------------------------------------------------------------
# Postselected DL passes
# ---------------------------------------------------------------------------

def dl_pass(psi, terms, layers, n):
    """One pass of the DL operator, applied as postselected measurements of
    {I - Pi_a, Pi_a}. Returns the normalized state and the probability that
    every measurement in the pass accepted."""
    prob = 1.0
    for layer in layers:
        for a in layer:
            h, supp = terms[a]
            rejected = apply_local(h, supp, psi, n)
            accepted = psi - rejected
            p = np.vdot(accepted, accepted).real
            prob *= p
            psi = accepted / np.sqrt(p)
    return psi, prob


def run_path(path, T, passes_per_step=1):
    """Discretized adiabatic path with T steps. Returns the total success
    probability and the final fidelity with the target ground state."""
    n = path.n
    psi = np.ones(2 ** n, dtype=complex) / np.sqrt(2 ** n)
    total = 1.0
    for t in range(1, T + 1):
        terms = path.terms(t / T)
        for _ in range(passes_per_step):
            psi, p = dl_pass(psi, terms, path.layers, n)
            total *= p
    fid = abs(np.vdot(path.ground_state(1.0), psi)) ** 2
    return total, fid


def run_direct(path, passes):
    """No path: apply DL passes of H(1) directly to |+>^n."""
    n = path.n
    psi = np.ones(2 ** n, dtype=complex) / np.sqrt(2 ** n)
    terms = path.terms(1.0)
    gs = path.ground_state(1.0)
    total, fids, probs = 1.0, [abs(np.vdot(gs, psi)) ** 2], [1.0]
    for _ in range(passes):
        psi, p = dl_pass(psi, terms, path.layers, n)
        total *= p
        fids.append(abs(np.vdot(gs, psi)) ** 2)
        probs.append(total)
    return np.array(fids), np.array(probs)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 11, "legend.fontsize": 9,
        "axes.linewidth": 0.8,
    })


def make_instance(n, seed, k=3, ratio=1.4):
    """Draw a path instance whose endpoint and midpoint ground states are
    unique (checked numerically)."""
    rng = np.random.default_rng(seed)
    m = int(round(ratio * n))
    while True:
        path = Path(n, m, k, rng)
        ok = True
        for s in (0.0, 0.5, 1.0):
            e0, gap = spectral_gap(path.terms(s), n)
            if abs(e0) > 1e-8 or gap < 1e-3:
                ok = False
                break
        if ok:
            return path


def experiment_gap_and_steps(n=10, seed=1):
    path = make_instance(n, seed)
    s_grid = np.linspace(0, 1, 21)
    gaps = [spectral_gap(path.terms(s), n)[1] for s in s_grid]

    Ts = [1, 2, 4, 8, 16, 32, 64, 128]
    res = [run_path(path, T) for T in Ts]
    probs = np.array([r[0] for r in res])
    fids = np.array([r[1] for r in res])

    fid_direct, prob_direct = run_direct(path, passes=128)

    print(f"[n={n}] m={len(path.supports)}, layers={len(path.layers)}, "
          f"min gap={min(gaps):.3f}")
    for T, p, f in zip(Ts, probs, fids):
        print(f"  path T={T:4d}: success={p:.4f}  fidelity={f:.6f}")
    print(f"  direct: initial overlap={fid_direct[0]:.4f}, "
          f"success after 128 passes={prob_direct[-1]:.4f}, "
          f"fidelity={fid_direct[-1]:.6f}")
    return path, s_grid, gaps, Ts, probs, fids, fid_direct, prob_direct


def experiment_scaling(ns=(4, 6, 8, 10, 12, 14), seeds=range(12), T=64):
    """Success probability vs n: adiabatic path (T steps) vs direct
    postselection, whose success probability is bounded by the initial
    overlap |<g(1)|+^n>|^2."""
    rows = []
    for n in ns:
        path_p, direct_p = [], []
        for seed in seeds:
            path = make_instance(n, 100 * n + seed)
            p, _ = run_path(path, T)
            path_p.append(p)
            _, probs = run_direct(path, passes=64)
            direct_p.append(probs[-1])
        gmean = lambda x: float(np.exp(np.mean(np.log(x))))
        rows.append((n, gmean(path_p), gmean(direct_p)))
        print(f"  n={n:2d}: path success (geo. mean)={rows[-1][1]:.4f}, "
              f"direct success (geo. mean)={rows[-1][2]:.2e}")
    return rows


if __name__ == "__main__":
    style()

    (path, s_grid, gaps, Ts, probs, fids,
     fid_direct, prob_direct) = experiment_gap_and_steps()

    # Figure 1: DL contraction (direct postselection on H(1))
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    it = np.arange(len(fid_direct[:11]))
    ax.plot(it, fid_direct[:11], "o-", color="black", ms=4, lw=1.2,
            label="overlap with ground state")
    ax.plot(it, prob_direct[:11], "s--", color="tab:red", ms=4, lw=1.2,
            label="cumulative success probability")
    ax.set_xlabel(r"DL passes $t$")
    ax.set_ylim(0, 1.05)
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.legend(loc="center right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/overlap_vs_iterations.png", dpi=300)

    # Figure 2: adiabatic path, success probability vs number of steps
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axes[0]
    ax.plot(s_grid, gaps, "o-", color="black", ms=3, lw=1.2)
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"spectral gap $\Delta(s)$")
    ax.set_ylim(0, None)
    ax.grid(True, lw=0.4, alpha=0.5)
    ax = axes[1]
    ax.semilogx(Ts, probs, "o-", color="tab:red", ms=4, lw=1.2,
                label="success probability")
    ax.semilogx(Ts, fids, "s--", color="black", ms=4, lw=1.2,
                label="final fidelity")
    ax.axhline(fid_direct[0], color="gray", lw=0.8, ls=":",
               label=r"$|\langle g(1)|{+}^{n}\rangle|^2$")
    ax.set_xlabel(r"number of path steps $T$")
    ax.set_ylim(0, 1.05)
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.legend(loc="lower right")
    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/adiabatic_path.png", dpi=300)

    # Figure 3: scaling with n
    print("Scaling with n:")
    rows = experiment_scaling()
    ns = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.semilogy(ns, [r[1] for r in rows], "o-", color="tab:red", ms=4,
                lw=1.2, label=r"adiabatic path ($T=64$)")
    ax.semilogy(ns, [r[2] for r in rows], "s--", color="black", ms=4,
                lw=1.2, label="direct postselection")
    ax.set_xlabel(r"number of qubits $n$")
    ax.set_ylabel("success probability")
    ax.grid(True, which="both", lw=0.4, alpha=0.5)
    ax.legend(loc="lower left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/success_vs_n.png", dpi=300)
