# Quantum-annealing shortest path (travel time)

Computes the minimum travel-time path between a source and a receiver in a 2-D
velocity model, formulated as a QUBO problem and solved with D-Wave samplers
(classical simulators or a real QPU). The result is compared with the exact
Dijkstra solution.

The QUBO formulation is the *undirected edge-based approach* of

> T. Krauss and J. McCollum, "Solving the Network Shortest Path Problem on a
> Quantum Annealer", *IEEE Trans. Quantum Eng.*, vol. 1, 2020,
> doi:10.1109/TQE.2020.3021921 (Sec. VI, eqs. 13, 17–19).

## Files

| File | Content |
|------|---------|
| `shortest_path.py` | Main script: grid, QUBO, optimization, figure |
| `velocity_model.py` | Velocity model: `velocity(x, y)` and `gradient(x, y)` |

## Requirements

```bash
pip install numpy scipy matplotlib dimod dwave-samplers pygmsh gmsh
pip install dwave-system dwave-networkx minorminer   # only for qpu / qpu-sim
```

`pygmsh`/`gmsh` are needed only for irregular grids. On Linux, gmsh may also
require system libraries such as `libglu1-mesa`, `libxft2`, `libxcursor1`,
`libxinerama1`.

## Usage

```bash
python shortest_path.py --grid regular --n 50 --source 0 0 --receiver 1 1

python shortest_path.py --grid irregular --n 60 \
       --source 0.05 0.1 --receiver 0.9 0.8 \
       --algorithm pimc --cycles 20 --sweeps 20000
```

`python shortest_path.py --help` lists all options.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `velocity_model.py` | File defining `velocity` and `gradient` |
| `--grid` | `regular` | `regular` or `irregular` |
| `--n` | `50` | Approximate number of grid nodes |
| `--no-diagonals` | off | Regular grid: omit cell diagonals |
| `--density-floor` | `0.1` | Irregular grid: minimum node density (fraction of the maximum) |
| `--source XS YS` | `0 0` | Source coordinates |
| `--receiver XR YR` | `1 1` | Receiver coordinates |
| `--algorithm` | `sa` | `sa`, `pimc`, `tabu`, `sd`, `qpu`, `qpu-sim` (see below) |
| `--cycles` | `10` | Number of sampler calls |
| `--reads` | `10` | Samples per call |
| `--sweeps` | `10000` | Sweeps per read (`sa`, `pimc`) |
| `--beta-range BMIN BMAX` | automatic | Inverse temperature range (`sa`, `pimc`) |
| `--chain-strength` | automatic | `qpu`, `qpu-sim`: chain coupling strength |
| `--annealing-time` | QPU default | `qpu`: annealing time per read (µs) |
| `--postprocess` | off | Refine each sample by greedy descent on the logical QUBO (recommended with `qpu`, `qpu-sim`) |
| `--k-i` | `0.3` | Inner-node penalty, in units of d_sr / v_min |
| `--k-st` | `2.5` | Source/receiver penalty, relative to the inner-node one |
| `--seed` | none | Random seed (incremented at each cycle) |
| `--output` | `FIG/shortest_path.png` | Output figure |
| `--timing-json` | none | Save parameters, results and timing to a JSON file |

Options not supported by the selected algorithm are ignored.

### Algorithms

| Name | Sampler |
|------|---------|
| `sa` | `SimulatedAnnealingSampler` (dwave-samplers) |
| `pimc` | `PathIntegralAnnealingSampler` — path-integral Monte Carlo simulation of quantum annealing |
| `tabu` | `TabuSampler` |
| `sd` | `SteepestDescentSolver` (greedy local descent) |
| `qpu` | `LazyFixedEmbeddingComposite(DWaveSampler())` — real D-Wave hardware, requires a Leap API token |
| `qpu-sim` | Emulated Advantage QPU: simulated annealing restricted to the Pegasus P16 topology, with minor embedding — no token needed |

For `qpu` and `qpu-sim` the embedding is computed at the first cycle and
reused, and the script reports the number of physical qubits, the chain
lengths and the fraction of broken chains.

**About `qpu-sim`.** It reproduces what the hardware imposes on the problem
(embedding overhead, chains, chain strength, chain breaks), so it is useful to
test the full QPU workflow and to estimate the qubit count before using real
QPU time. It does *not* reproduce the quality of the QPU results: classical
single-spin-flip annealing performs poorly on embedded problems because chains
of coupled qubits freeze and rarely flip together, so expect far fewer valid
paths than with `sa` on the logical problem. `--postprocess` (greedy descent on
the unembedded samples) and more sweeps recover some valid paths.

**Trial accounts.** Self-sign-up Leap trial accounts do not provide an API
token, so `qpu` cannot be used with them; use `qpu-sim` or the classical
samplers instead.

## Method

### Domain and grids

The model domain is the unit square [0, 1] × [0, 1].

- **Regular grid**: M × M nodes, with M = round(√N). Edges connect horizontal
  and vertical neighbours and, unless `--no-diagonals` is given, both
  diagonals of each cell.
- **Irregular grid**: triangular mesh generated with gmsh, with node density

  ρ(x, y) = f + (1 − f) · |∇v| / max|∇v|

  where f is `--density-floor` (it prevents infinitely large elements where
  the gradient vanishes). The local element size is h ∝ 1/√ρ, and its scale is
  calibrated iteratively to obtain about N nodes.

Source and receiver are snapped to the nearest grid node.

Each edge cost is its travel time: length / mean velocity at its two end points.

### QUBO

One binary variable per node (x_i = 1 if node i is on the path) and one per
edge (x_ij = 1 if the edge is on the path), i.e. |V| + |E| variables:

H = Σ c_ij x_ij + α_st [H_s + H_t] + α_i Σ_{i≠s,t} (2 x_i − Σ_j x_ij)²

with H_s = −x_s + (x_s − Σ_j x_sj)² (same for H_t). The penalties force degree
1 at source and receiver, degree 2 at inner path nodes and 0 elsewhere.

**Penalty scale.** The paper uses α > Σ|c_ij| (sum of all edge costs), which
on a mesh is very large and degrades the annealing. Here
α_i = k_i · d_sr / v_min (straight-line distance divided by the minimum
velocity) and α_st = k_st · α_i. Too large penalties make the sampler satisfy
the constraints while ignoring the travel time; too small penalties produce
invalid samples. `--k-i` and `--k-st` control this trade-off.

### Validation

The penalties are soft, so the sampler can return configurations that are not
paths. Each sample is checked (node degrees and a single connected chain of
edges); among the **valid** samples, the one with the lowest travel time is
kept. The ratio of valid samples to total samples is printed and shown in the
figure title.

## Output

The console reports grid size, number of QUBO variables, snapped
source/receiver nodes, the progress of each cycle, and the QUBO and Dijkstra
travel times.

The figure shows:

- the velocity model as a colour scale;
- all grid nodes and connections (thin lines);
- source (star) and receiver (triangle);
- the best valid path of each cycle (thin semi-transparent cyan lines;
  overlapping lines look darker, so recurring paths stand out);
- the optimized QUBO path, best over all cycles (thick red line);
- the exact Dijkstra path (white dashed line), for comparison.

### Timing statistics

At the end of the run the script prints:

- wall-clock time of each phase (model loading, grid, QUBO construction,
  sampler setup, sampling + validation, Dijkstra, plot) and its percentage;
- per-cycle sampler time (mean, std, min, max) and time per read;
- for `sa` and `pimc`, the time per sweep per variable (cost of one spin update);
- the QPU access time, when reported by the sampler (`qpu`);
- the time spent validating samples;
- when the first valid path and the best path were found (cycle, read and
  cumulative sampling time).

With `--timing-json` all of this is also saved, together with the run
parameters and results, which makes it easy to collect scaling studies over
many runs (e.g. time vs. number of variables).

## Custom velocity models

`velocity_model.py` must define two functions accepting scalars and NumPy
arrays:

```python
def velocity(x, y):        # velocity > 0
    ...

def gradient(x, y):        # returns (dv/dx, dv/dy), or |grad v|
    ...
```

The provided example is a Gaussian high-velocity anomaly
(v = 1 + 2 exp(−r²/0.2²), centred at (0.5, 0.5)) with its analytical gradient.

## Notes and limitations

- Problem size grows quickly: |V| + |E| variables (about 5M² for a regular
  grid with diagonals). Simulated samplers handle a few hundred variables; a
  real QPU needs minor embedding, which multiplies the physical qubits.
- For small problems `tabu` usually finds the exact path; `sa` gets close with
  tuned penalties; `pimc` needs many sweeps.
- Dijkstra solves the same problem exactly in O(|E| + |V| log |V|): the purpose
  of this code is to study the behaviour of (quantum) annealing on a physically
  meaningful problem, not to outperform classical algorithms.
