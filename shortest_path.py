"""
Source-receiver shortest (minimum travel time) path as a QUBO problem.

QUBO formulation: "undirected edge-based approach" of
Krauss & McCollum (2020), IEEE Trans. Quantum Eng., Sec. VI (eqs. 13, 17-19).

Domain: normalized square [0, 1] x [0, 1].
Velocity model: functions velocity(x, y) and gradient(x, y) loaded from an
external file (default: velocity_model.py).

Examples
--------
    python shortest_path.py --grid regular   --n 50  --source 0 0 --receiver 1 1
    python shortest_path.py --grid irregular --n 60  --source 0.1 0.2 --receiver 0.9 0.7 \
                            --algorithm pimc --cycles 20 --sweeps 20000

QUBO variables (integer labels):
    0 .. nnodes-1               -> x_i   (node i on the path)
    nnodes .. nnodes+nedges-1   -> x_ij  (edge k = (i, j) on the path)
"""

import argparse
import importlib.util
import json
import os
import time
from contextlib import contextmanager

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import dimod
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


# =============================================================================
# Velocity model
# =============================================================================

def load_velocity_model(path):
    """Load velocity(x, y) and gradient(x, y) from a Python file."""
    spec = importlib.util.spec_from_file_location("velocity_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("velocity", "gradient"):
        if not hasattr(module, name):
            raise AttributeError(f"{path} must define a function '{name}(x, y)'")
    return module.velocity, module.gradient


def gradient_magnitude(gradient, x, y):
    """|grad v|; gradient() may return (dvx, dvy) or the magnitude directly."""
    g = gradient(x, y)
    if isinstance(g, (tuple, list)) and len(g) == 2:
        return np.hypot(g[0], g[1])
    return np.abs(g)


# =============================================================================
# Grids
# =============================================================================

def edge_costs(nodes, edges, velocity):
    """Travel time of each edge: length / mean velocity at its end points."""
    v = velocity(nodes[:, 0], nodes[:, 1])
    length = np.linalg.norm(nodes[edges[:, 1]] - nodes[edges[:, 0]], axis=1)
    return length / (0.5 * (v[edges[:, 0]] + v[edges[:, 1]]))


def regular_grid(n_target, diagonals=True):
    """
    M x M regular grid with M*M as close as possible to n_target.
    Edges: horizontal, vertical and (optionally) both diagonals of each cell.
    """
    m = max(2, int(round(np.sqrt(n_target))))
    i, j = np.meshgrid(np.arange(m), np.arange(m))     # shape (m, m)
    idx = i + j * m
    nodes = np.column_stack([i.ravel(), j.ravel()]) / (m - 1.0)

    pairs = [
        np.column_stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()]),   # horizontal
        np.column_stack([idx[:-1, :].ravel(), idx[1:, :].ravel()]),   # vertical
    ]
    if diagonals:
        pairs += [
            np.column_stack([idx[:-1, :-1].ravel(), idx[1:, 1:].ravel()]),  # "/"
            np.column_stack([idx[1:, :-1].ravel(), idx[:-1, 1:].ravel()]),  # "\"
        ]
    edges = np.sort(np.vstack(pairs), axis=1).astype(np.int64)
    return nodes, edges


def _gmsh_triangulation(size_fun):
    """Triangulate the unit square with local element size size_fun(x, y)."""
    import pygmsh   # imported here: only needed for irregular grids

    with pygmsh.geo.Geometry() as geom:
        geom.add_polygon([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        geom.set_mesh_size_callback(
            lambda dim, tag, x, y, z, lc: float(size_fun(x, y)))
        mesh = geom.generate_mesh()

    tri = np.vstack([c.data for c in mesh.cells if c.type == "triangle"])
    tri = tri.astype(np.int64)

    # Keep only the points used by triangles and renumber them
    used, tri = np.unique(tri, return_inverse=True)
    tri = tri.reshape(-1, 3)
    nodes = np.asarray(mesh.points)[used, :2]

    edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    return nodes, edges


def irregular_grid(n_target, gradient, density_floor=0.1, n_iter=4):
    """
    Triangular mesh with about n_target nodes and node density proportional
    to |grad v|. A floor (fraction of the maximum density) avoids infinitely
    large elements where the gradient vanishes:

        rho(x, y) = floor + (1 - floor) * |grad v| / max|grad v|
        h(x, y)   = C / sqrt(rho)          (density ~ 1 / h^2)

    The constant C is calibrated iteratively to reach n_target nodes.
    """
    g = np.linspace(0.0, 1.0, 201)
    gx, gy = np.meshgrid(g, g)
    gmax = gradient_magnitude(gradient, gx, gy).max()
    if gmax <= 0:                                   # constant velocity
        gmax = 1.0

    def rho(x, y):
        w = np.clip(gradient_magnitude(gradient, x, y) / gmax, 0.0, 1.0)
        return density_floor + (1.0 - density_floor) * w

    # First guess: an equilateral mesh of size h has ~2/(sqrt(3) h^2) nodes/area
    mean_rho = rho(gx, gy).mean()
    c = np.sqrt(2.0 / np.sqrt(3.0) * mean_rho / n_target)

    for _ in range(n_iter):
        nodes, edges = _gmsh_triangulation(
            lambda x, y: min(c / np.sqrt(rho(x, y)), 0.5))
        ratio = len(nodes) / n_target
        if abs(ratio - 1.0) < 0.05:
            break
        c *= np.sqrt(ratio)
    return nodes, edges


def nearest_node(nodes, point):
    return int(np.argmin(np.linalg.norm(nodes - np.asarray(point), axis=1)))


# =============================================================================
# QUBO (Krauss & McCollum 2020, Sec. VI)
# =============================================================================

def build_bqm(nnodes, edges, costs, source, receiver, alpha_i, alpha_st):
    """
    H = Hc + alpha_st * (Hs + Ht) + alpha_i * sum_{i != s,t} Hi

        Hc = sum_k c_k x_k                    (eq. 13)
        Hs = -x_s + (x_s - sum_j x_sj)^2      (eq. 17; Ht analogous, eq. 18)
        Hi = (2 x_i - sum_j x_ij)^2           (eq. 19)

    (For binary variables x^2 = x.)
    """
    nedges = len(edges)
    incident = [[] for _ in range(nnodes)]
    for k, (i, j) in enumerate(edges):
        incident[i].append(nnodes + k)
        incident[j].append(nnodes + k)

    bqm = dimod.BinaryQuadraticModel("BINARY")
    bqm.add_variables_from((v, 0.0) for v in range(nnodes + nedges))

    for k in range(nedges):                                   # Hc
        bqm.add_linear(nnodes + k, costs[k])

    for i in range(nnodes):
        endpoint = i in (source, receiver)
        a, weight = (1, alpha_st) if endpoint else (2, alpha_i)
        terms = [(i, a)] + [(v, -1) for v in incident[i]]
        bqm.add_linear_equality_constraint(terms, weight, constant=0)
        if endpoint:
            bqm.add_linear(i, -alpha_st)                      # -x_s, -x_t
    return bqm


def is_valid_path(nnodes, edges, source, receiver, x):
    """
    True if x encodes a simple path from source to receiver:
    correct node degrees (1 at the end points, 2 at the inner nodes,
    0 elsewhere) and a single connected chain of edges.
    """
    x_nodes, x_edges = x[:nnodes], x[nnodes:]
    selected = edges[np.flatnonzero(x_edges)]
    if not (x_nodes[source] and x_nodes[receiver]) or len(selected) == 0:
        return False
    if not np.all(x_nodes[selected]):
        return False

    degree = np.bincount(selected.ravel(), minlength=nnodes)
    expected = 2 * x_nodes.astype(np.int64)
    expected[source] = expected[receiver] = 1
    if not np.array_equal(degree, expected):
        return False

    neighbors = {}
    for i, j in selected:
        neighbors.setdefault(i, []).append(j)
        neighbors.setdefault(j, []).append(i)
    prev, cur, n_steps = -1, source, 0
    while cur != receiver:
        prev, cur = cur, next(n for n in neighbors[cur] if n != prev)
        n_steps += 1
    return n_steps == len(selected)


# =============================================================================
# Timing
# =============================================================================

class PhaseTimer:
    """Wall-clock time of the main phases of the run."""

    def __init__(self):
        self.phases = {}

    @contextmanager
    def __call__(self, name):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.phases[name] = self.phases.get(name, 0.0) + time.perf_counter() - t0

    @property
    def total(self):
        return sum(self.phases.values())


def format_seconds(t):
    if t < 1e-3:
        return f"{t * 1e6:8.1f} us"
    if t < 1.0:
        return f"{t * 1e3:8.2f} ms"
    return f"{t:8.3f} s "


def print_embedding_report(sampler, stats):
    """Embedding size and chain breaks (qpu and qpu-sim only)."""
    embedding = getattr(sampler, "embedding", None)
    if not embedding:
        return
    chains = [len(c) for c in embedding.values()]
    print("\nEmbedding")
    print(f"  logical variables: {len(chains)}  physical qubits: {sum(chains)}")
    print(f"  chain length: mean {np.mean(chains):.2f}  max {max(chains)}")
    cbf = np.asarray(stats["chain_break_fraction"])
    if len(cbf):
        print(f"  broken chains per sample: mean {100 * cbf.mean():.2f}%  "
              f"samples with broken chains {100 * np.mean(cbf > 0):.1f}%")


def print_timing_report(timer, stats, n_reads, n_sweeps, nvar, algorithm):
    total = timer.total
    print("\nComputation time")
    print("  Phase                     time        %")
    for name, t in timer.phases.items():
        print(f"  {name:<22}{format_seconds(t)}  {100 * t / total:6.1f}")
    print(f"  {'total':<22}{format_seconds(total)}")

    ct = np.asarray(stats["cycle_times"])
    if len(ct) == 0:
        return
    print("\nSampler calls (cycles)")
    print(f"  per cycle:  mean {format_seconds(ct.mean())}  std {format_seconds(ct.std())}"
          f"  min {format_seconds(ct.min())}  max {format_seconds(ct.max())}")
    per_read = ct.sum() / max(stats["n_total"], 1)
    print(f"  per read:   {format_seconds(per_read)}")
    if algorithm in ("sa", "pimc"):
        ns = per_read / (n_sweeps * nvar) * 1e9
        print(f"  per sweep x variable (spin update): {ns:.2f} ns")
    if stats["qpu_access_time"] > 0:
        print(f"  QPU access time (from sampleset info): "
              f"{format_seconds(stats['qpu_access_time'])}")
    if stats["postprocess_time"] > 0:
        print(f"  greedy post-processing: {format_seconds(stats['postprocess_time'])}"
              f"  (valid samples before: {stats['n_valid_raw']}, "
              f"after: {stats['n_valid']})")
    print(f"  validation of samples: {format_seconds(stats['validation_time'])}")

    for label, key in (("first valid path", "first_valid"),
                       ("best path", "best_found")):
        info = stats[key]
        if info is None:
            print(f"  {label}: not found")
        else:
            print(f"  {label}: cycle {info['cycle']}, read {info['read']} "
                  f"(after {format_seconds(info['sampling_time'])} of sampling)")


# =============================================================================
# Samplers
# =============================================================================

ALGORITHMS = ("sa", "pimc", "tabu", "sd", "qpu", "qpu-sim")


def pegasus_sampler(child, size=16):
    """
    Emulated D-Wave Advantage: the child sampler is restricted to the Pegasus
    P16 topology (5640 qubits), and the logical problem is minor-embedded on
    it with chains of physical qubits. This reproduces embedding overhead,
    chain strength effects and chain breaks without a Leap API token.
    The embedding is computed at the first call and then reused.
    """
    import dwave_networkx as dnx
    from dimod import StructureComposite
    from dwave.system import LazyFixedEmbeddingComposite

    graph = dnx.pegasus_graph(size)
    return LazyFixedEmbeddingComposite(
        StructureComposite(child, list(graph.nodes), list(graph.edges)))


def make_sampler(name):
    if name == "sa":
        from dwave.samplers import SimulatedAnnealingSampler
        return SimulatedAnnealingSampler()
    if name == "pimc":
        from dwave.samplers import PathIntegralAnnealingSampler
        return PathIntegralAnnealingSampler()
    if name == "tabu":
        from dwave.samplers import TabuSampler
        return TabuSampler()
    if name == "sd":
        from dwave.samplers import SteepestDescentSolver
        return SteepestDescentSolver()
    if name == "qpu":   # requires dwave-system and a configured Leap token
        from dwave.system import DWaveSampler, LazyFixedEmbeddingComposite
        return LazyFixedEmbeddingComposite(DWaveSampler())
    if name == "qpu-sim":   # no token needed: SA on an emulated Pegasus QPU
        from dwave.samplers import SimulatedAnnealingSampler
        return pegasus_sampler(SimulatedAnnealingSampler())
    raise ValueError(f"Unknown algorithm '{name}'. Choose from {ALGORITHMS}")


def greedy_postprocess(bqm, samples):
    """
    Steepest-descent refinement of each sample on the logical QUBO: single
    variable flips are applied while they lower the energy. Common practice
    with QPU results, where broken or frozen chains leave samples close to,
    but not at, a local minimum.
    """
    from dwave.samplers import SteepestDescentSolver
    labels = list(range(bqm.num_variables))
    result = SteepestDescentSolver().sample(bqm, initial_states=(samples, labels))
    order = [result.variables.index(v) for v in labels]
    return result.record.sample[:, order]


def run_sampler(bqm, sampler, n_cycles, is_valid, cost, postprocess=False,
                **kwargs):
    """
    Call the sampler n_cycles times, keeping the cheapest valid sample.
    Keyword arguments not supported by the sampler are dropped.
    If postprocess is True, each sample is refined by greedy descent
    before validation.

    Returns (best_x, best_cost, stats, cycle_best):
      stats      counts and timing: wall time of each sampler call, validation
                 time, QPU access time (if reported by the sampler), and when
                 the first valid / best path was found;
      cycle_best list with the best valid sample of each cycle (None if the
                 cycle produced no valid sample).
    """
    kwargs = {k: v for k, v in kwargs.items()
              if k in sampler.parameters and v is not None}
    nvar = bqm.num_variables
    best_x, best_cost = None, np.inf
    stats = dict(n_valid=0, n_total=0, cycle_times=[], validation_time=0.0,
                 qpu_access_time=0.0, first_valid=None, best_found=None,
                 chain_break_fraction=[], n_valid_raw=0, postprocess_time=0.0)
    sampling_time = 0.0
    cycle_best = []

    seed = kwargs.pop("seed", None)
    for cycle in range(1, n_cycles + 1):
        if seed is not None and "seed" in sampler.parameters:
            kwargs["seed"] = seed + cycle        # different but reproducible runs

        t0 = time.perf_counter()
        sampleset = sampler.sample(bqm, **kwargs)
        sampleset.resolve()                      # QPU/async samplers
        dt = time.perf_counter() - t0
        stats["cycle_times"].append(dt)
        sampling_time += dt
        timing = sampleset.info.get("timing", {})
        stats["qpu_access_time"] += timing.get("qpu_access_time", 0.0) * 1e-6
        if "chain_break_fraction" in sampleset.record.dtype.names:
            stats["chain_break_fraction"] += \
                sampleset.record.chain_break_fraction.tolist()

        order = [sampleset.variables.index(v) for v in range(nvar)]
        samples = sampleset.record.sample[:, order]
        if postprocess:
            stats["n_valid_raw"] += sum(bool(is_valid(x)) for x in samples)
            t0 = time.perf_counter()
            samples = greedy_postprocess(bqm, samples)
            stats["postprocess_time"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        cyc_x, cyc_cost = None, np.inf
        for read, x in enumerate(samples, start=1):
            stats["n_total"] += 1
            if not is_valid(x):
                continue
            stats["n_valid"] += 1
            where = dict(cycle=cycle, read=read, sampling_time=sampling_time)
            if stats["first_valid"] is None:
                stats["first_valid"] = where
            c = cost(x)
            if c < cyc_cost:
                cyc_x, cyc_cost = x.copy(), c
            if c < best_cost:
                best_x, best_cost = x.copy(), c
                stats["best_found"] = where
        stats["validation_time"] += time.perf_counter() - t0
        cycle_best.append(cyc_x)

        raw = f" (raw {stats['n_valid_raw']})" if postprocess else ""
        print(f"  cycle {cycle}/{n_cycles}: valid {stats['n_valid']}{raw}/"
              f"{stats['n_total']}, best time {best_cost:.4f}  "
              f"({format_seconds(dt).strip()})")
    return best_x, best_cost, stats, cycle_best


# =============================================================================
# Reference solution
# =============================================================================

def dijkstra_path(nnodes, edges, costs, source, receiver):
    """Exact travel time and node sequence (for comparison)."""
    graph = coo_matrix((costs, (edges[:, 0], edges[:, 1])), shape=(nnodes, nnodes))
    dist, pred = dijkstra(graph, directed=False, indices=source,
                          return_predecessors=True)
    path = [receiver]
    while path[-1] != source:
        path.append(pred[path[-1]])
    return dist[receiver], path[::-1]


# =============================================================================
# Plot
# =============================================================================

def plot_result(nodes, edges, velocity, source, receiver, qubo_edges=None,
                reference_path=None, cycle_edges=(), title="",
                filename="shortest_path.png"):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Velocity model
    g = np.linspace(0.0, 1.0, 300)
    gx, gy = np.meshgrid(g, g)
    im = ax.imshow(velocity(gx, gy), extent=(0, 1, 0, 1), origin="lower",
                   cmap="viridis", interpolation="bilinear")
    fig.colorbar(im, ax=ax, label="Velocity")

    # All connections (thin lines) and nodes
    ax.add_collection(LineCollection(nodes[edges], colors="k", linewidths=0.4,
                                     alpha=0.35))
    ax.plot(nodes[:, 0], nodes[:, 1], "o", color="k", ms=2.5)

    # Best path of each cycle (thin cyan lines; overlaps show recurring paths)
    for k, ce in enumerate(cycle_edges):
        ax.add_collection(LineCollection(nodes[ce], colors="cyan", linewidths=1.0,
                                         alpha=0.5, zorder=4,
                                         label="Cycle results" if k == 0 else None))

    # Exact reference path (dashed)
    if reference_path is not None:
        ax.plot(nodes[reference_path, 0], nodes[reference_path, 1], "--",
                color="white", lw=1.5, label="Dijkstra", zorder=6)

    # Optimized path (thick line)
    if qubo_edges is not None and len(qubo_edges):
        ax.add_collection(LineCollection(nodes[qubo_edges], colors="red",
                                         linewidths=3.0, label="QUBO",
                                         zorder=5))

    ax.plot(*nodes[source], "*", color="magenta", mec="k", ms=18, label="Source",
            clip_on=False, zorder=10)
    ax.plot(*nodes[receiver], "v", color="cyan", mec="k", ms=13, label="Receiver",
            clip_on=False, zorder=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    fig.tight_layout()

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    fig.savefig(filename, dpi=150)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="velocity_model.py",
                   help="file defining velocity(x, y) and gradient(x, y)")
    p.add_argument("--grid", choices=("regular", "irregular"), default="regular")
    p.add_argument("--n", type=int, default=50, help="approximate number of nodes")
    p.add_argument("--no-diagonals", action="store_true",
                   help="regular grid: omit the cell diagonals")
    p.add_argument("--density-floor", type=float, default=0.1,
                   help="irregular grid: minimum density (fraction of the maximum)")
    p.add_argument("--source", type=float, nargs=2, default=(0.0, 0.0),
                   metavar=("XS", "YS"))
    p.add_argument("--receiver", type=float, nargs=2, default=(1.0, 1.0),
                   metavar=("XR", "YR"))
    p.add_argument("--algorithm", choices=ALGORITHMS, default="sa",
                   help="sa: simulated annealing, pimc: path-integral annealing, "
                        "tabu: tabu search, sd: steepest descent, qpu: D-Wave QPU, "
                        "qpu-sim: SA on an emulated Pegasus QPU (no token needed)")
    p.add_argument("--cycles", type=int, default=10, help="number of sampler calls")
    p.add_argument("--reads", type=int, default=10, help="samples per call")
    p.add_argument("--sweeps", type=int, default=10000,
                   help="sweeps per read (sa, pimc)")
    p.add_argument("--beta-range", type=float, nargs=2, default=None,
                   metavar=("BMIN", "BMAX"),
                   help="inverse temperature range (sa, pimc); default: automatic")
    p.add_argument("--chain-strength", type=float, default=None,
                   help="qpu, qpu-sim: chain coupling strength (default: Ocean's "
                        "uniform torque compensation)")
    p.add_argument("--annealing-time", type=float, default=None,
                   help="qpu: annealing time per read in microseconds")
    p.add_argument("--postprocess", action="store_true",
                   help="refine every sample by greedy descent on the logical "
                        "QUBO (recommended with qpu and qpu-sim)")
    p.add_argument("--k-i", type=float, default=0.3,
                   help="inner-node penalty, in units of d_sr / v_min")
    p.add_argument("--k-st", type=float, default=2.5,
                   help="source/receiver penalty, relative to the inner-node one")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", default="FIG/shortest_path.png")
    p.add_argument("--timing-json", default=None,
                   help="save run parameters, results and timing to a JSON file")
    return p.parse_args()


def main():
    args = parse_args()
    timer = PhaseTimer()
    with timer("load model"):
        velocity, gradient = load_velocity_model(args.model)

    # --- Grid ---------------------------------------------------------------
    with timer("grid"):
        if args.grid == "regular":
            nodes, edges = regular_grid(args.n, diagonals=not args.no_diagonals)
        else:
            nodes, edges = irregular_grid(args.n, gradient, args.density_floor)
        costs = edge_costs(nodes, edges, velocity)
    nnodes, nedges = len(nodes), len(edges)
    print(f"Grid: {args.grid}  nodes: {nnodes}  edges: {nedges}  "
          f"QUBO variables: {nnodes + nedges}")

    # --- Source and receiver -------------------------------------------------
    source = nearest_node(nodes, args.source)
    receiver = nearest_node(nodes, args.receiver)
    if source == receiver:
        raise ValueError("Source and receiver map to the same grid node")
    for label, req, k in (("Source  ", args.source, source),
                          ("Receiver", args.receiver, receiver)):
        print(f"{label} ({req[0]:.3f}, {req[1]:.3f}) -> node {k} "
              f"({nodes[k, 0]:.3f}, {nodes[k, 1]:.3f})")

    # --- QUBO ---------------------------------------------------------------
    # Penalty scale: straight-ray travel time at the minimum velocity of the
    # grid, a rough upper estimate of the shortest path cost. (The paper's
    # bound, sum of all edge costs, is far too large on a mesh.)
    v_min = velocity(nodes[:, 0], nodes[:, 1]).min()
    t_scale = np.linalg.norm(nodes[receiver] - nodes[source]) / v_min
    alpha_i = args.k_i * t_scale
    alpha_st = args.k_st * alpha_i
    with timer("QUBO construction"):
        bqm = build_bqm(nnodes, edges, costs, source, receiver, alpha_i, alpha_st)

    # --- Optimization -------------------------------------------------------
    with timer("sampler setup"):
        sampler = make_sampler(args.algorithm)
    print(f"Algorithm: {args.algorithm}  cycles: {args.cycles}  reads: {args.reads}")
    with timer("sampling + validation"):
        x, t_qubo, stats, cycle_best = run_sampler(
            bqm, sampler, args.cycles,
            is_valid=lambda x: is_valid_path(nnodes, edges, source, receiver, x),
            cost=lambda x: float(x[nnodes:] @ costs),
            num_reads=args.reads, num_sweeps=args.sweeps,
            beta_range=tuple(args.beta_range) if args.beta_range else None,
            beta_schedule_type="linear",
            postprocess=args.postprocess,
            chain_strength=args.chain_strength,
            annealing_time=args.annealing_time,
            seed=args.seed)
    n_valid, n_total = stats["n_valid"], stats["n_total"]

    # --- Results ------------------------------------------------------------
    with timer("Dijkstra"):
        t_exact, ref_path = dijkstra_path(nnodes, edges, costs, source, receiver)
    print(f"\nDijkstra travel time: {t_exact:.4f}")
    if x is None:
        print(f"QUBO: no valid path in {n_total} samples")
        qubo_edges = None
        title = f"{args.algorithm.upper()}: no valid path  |  Dijkstra t = {t_exact:.4f}"
    else:
        print(f"QUBO travel time:     {t_qubo:.4f}  (ratio {t_qubo / t_exact:.3f}, "
              f"valid samples {n_valid}/{n_total})")
        qubo_edges = edges[np.flatnonzero(x[nnodes:])]
        title = (f"{args.algorithm.upper()} t = {t_qubo:.4f}  |  "
                 f"Dijkstra t = {t_exact:.4f}  |  valid {n_valid}/{n_total}")

    cycle_edges = [edges[np.flatnonzero(xc[nnodes:])]
                   for xc in cycle_best if xc is not None]
    with timer("plot"):
        plot_result(nodes, edges, velocity, source, receiver, qubo_edges, ref_path,
                    cycle_edges=cycle_edges,
                    title=f"{args.grid} grid, {nnodes} nodes\n{title}",
                    filename=args.output)
    print(f"Figure saved to {args.output}")

    # --- Timing statistics --------------------------------------------------
    print_embedding_report(sampler, stats)
    print_timing_report(timer, stats, args.reads, args.sweeps, nnodes + nedges,
                        args.algorithm)
    if args.timing_json:
        report = dict(
            parameters=vars(args),
            grid=dict(nodes=nnodes, edges=nedges, variables=nnodes + nedges),
            result=dict(qubo_time=None if x is None else t_qubo,
                        dijkstra_time=float(t_exact),
                        n_valid=n_valid, n_total=n_total),
            phases=timer.phases,
            sampling=stats,
        )
        os.makedirs(os.path.dirname(args.timing_json) or ".", exist_ok=True)
        with open(args.timing_json, "w") as f:
            json.dump(report, f, indent=2, default=float)
        print(f"Timing saved to {args.timing_json}")


if __name__ == "__main__":
    main()
