
# Systolic Recurrent Wave Network (SRWN)
# Systolic Recurrent Wave Network

Systolic Recurrent Wave Network (SRWN) is a grid-structured classifier designed for anytime inference. Each wave updates a 2D mesh of hidden states. Later columns are allowed more spatial compute, and later waves refine earlier outputs through recurrent right-neighbor reads.

This repository now separates the project into three layers:

- `README.md`: project entry point and run instructions
- `design.md`: current architecture, assumptions, and validation criteria
- `experiments.md`: experiment plan plus recorded results
- `redundant.md`: archived duplicated notes and alternative formulations from the original README

## Repository Contents

| File | Purpose |
|---|---|
| `systolic_net.py` | Original NumPy prototype aimed at MNIST training and C export |
| `srwn_experiment.py` | Focused PyTorch experiment runner for validating the core recurrent-wave claim |
| `design.md` | Clean design spec |
| `experiments.md` | Validation plan and measured outcomes |
| `redundant.md` | Preserved brainstorm and repeated material moved out of the main README |

## Current Validation Focus

The first experiment tests the claim that recurrent right-reads let earlier exits improve over successive waves.

Why this comes first:

1. If early columns do not improve when recurrence is enabled, the architecture is not delivering its intended anytime behavior.
2. Raw MNIST accuracy is a weak signal because a broken design can still score well enough to look plausible.
3. A synthetic parity task is cheap to run, isolates lateral communication, and exposes architectural flaws quickly.

## How To Run The Validation Experiment

The experiment script trains two versions of the same SRWN on 8-bit parity:

- recurrent: right-neighbor reads enabled across waves
- ablated: right-neighbor reads disabled

Run:

```bash
D:/apps/Python39/python.exe srwn_experiment.py --json
```

That command prints JSON with per-wave and per-column accuracy, monotonicity checks, and a simple confidence-halting summary.

## What To Read Next

- See [design.md](design.md) for the cleaned architecture and acceptance criteria.
- See [experiments.md](experiments.md) for the experiment plan, measured results, and flaws found.
- See [redundant.md](redundant.md) for the archived duplicated analysis from the original README.
### Node Design

Each node does:
```
h_out = σ(W_self · h_in_top + W_lat · h_in_left + b)
```
Where `h` is a small vector (16–64 floats). Each bottom node has a readout head:
```
logits_j = W_head · h_bottom_j
conf_j = 1 - H(softmax(logits_j))   # entropy as confidence
```

---

### Training Strategy

Multi-exit loss with position weighting:
```
L = Σ_j λ_j · CE(y, f_j(x))
```
- `λ_j` increasing rightward (reward accuracy at far right)
- Or uniform with [[Knowledge Distillation]]: left exits trained to match right exit's logits
- Key: **every bottom node must receive gradient** or left exits collapse to mean

Without this, left exits learn nothing useful — this is the #1 failure mode.

---

### Python Sketch

```python
import numpy as np

class SystolicNode:
    def __init__(self, d_in, d_out):
        self.W_top = np.random.randn(d_out, d_in) * 0.01
        self.W_lat = np.random.randn(d_out, d_in) * 0.01
        self.b = np.zeros(d_out)
    
    def forward(self, h_top, h_lat):
        return np.tanh(self.W_top @ h_top + self.W_lat @ h_lat + self.b)

class SystolicNet:
    def __init__(self, n_cols, depth, d_hidden, n_classes):
        self.depth = depth
        self.n_cols = n_cols
        # Triangular: col j has (depth - j) rows
        self.nodes = {
            (r, c): SystolicNode(d_hidden, d_hidden)
            for c in range(n_cols)
            for r in range(depth - c)
        }
        self.heads = {c: np.random.randn(n_classes, d_hidden) * 0.01 
                      for c in range(n_cols)}
    
    def forward(self, x_slices):
        # x_slices: list of n_cols input vectors
        activations = {}
        
        for r in range(self.depth):
            for c in range(self.n_cols - r):  # triangular bound
                h_top = x_slices[c] if r == 0 else activations.get((r-1, c), np.zeros_like(x_slices[c]))
                h_lat = x_slices[0]*0 if c == 0 else activations.get((r, c-1), np.zeros_like(x_slices[c]))
                activations[(r, c)] = self.nodes[(r, c)].forward(h_top, h_lat)
        
        # Read bottom nodes
        outputs = []
        for c in range(self.n_cols):
            bottom_r = self.depth - c - 1
            h = activations[(bottom_r, c)]
            logits = self.heads[c] @ h
            probs = np.exp(logits) / np.exp(logits).sum()
            conf = 1.0 - (-np.sum(probs * np.log(probs + 1e-9)) / np.log(len(probs)))
            outputs.append((probs, conf, c))
        
        return outputs
    
    def anytime_predict(self, x_slices, conf_threshold=0.9, budget=None):
        results = self.forward(x_slices)
        for probs, conf, col in results:
            if conf >= conf_threshold or (budget and col >= budget):
                return probs.argmax(), conf, col
        return results[-1][0].argmax(), results[-1][1], len(results)-1
```

---

### FPGA Assessment — **Yes, excellent fit**

| Concern | Reality |
|---|---|
| 2D mesh routing | Native to FPGA fabric |
| Multiply-accumulate | DSP48/DSP58 slices, 1 per node |
| Weight storage | BRAMs; 8-bit quant = ~4× more nodes |
| Anytime readout | Just a MUX on bottom row with confidence comparator |
| Pipelining | Each diagonal wavefront = one clock cycle |
| Clock-gating exits | Built-in: stop driving pipeline when threshold hit |

**Anti-diagonal wavefronts** are the key: nodes on the same anti-diagonal have no data dependencies → fully parallel. One new wavefront per clock. Total latency = `depth + n_cols - 1` cycles, not `depth * n_cols`.

Target: Artix-7 or iCE40 for prototype. Xilinx HLS or SpinalHDL would let you describe the mesh cleanly.

---

### Microcontroller

| Platform | Verdict |
|---|---|
| RP2040 (264KB SRAM) | 4×4 array, int8, barely — 2 cores helps |
| ESP32-S3 (AI extensions) | 8×8 plausible with SIMD |
| STM32H7 (FPU, 1MB SRAM) | 8×8 float32 feasible |
| Arduino Uno | No |

For RP2040: int8 weights, d_hidden=8, 4 columns → ~2KB weights, fits. Useful as proof-of-concept. Use core0 for inference, core1 for output polling.

---

### Success / Failure Spectrum

**Success:**
- Accuracy at column j monotonically increases with j
- Confidence scores are calibrated ([[Expected Calibration Error]] < 0.05)
- 50th-percentile input hits threshold at column 2 of 8 → 4× throughput gain
- Maps to FPGA with <500 DSP slices for useful task

**Failure:**
- Left exits all output uniform distribution (gradient starvation) — fix: distillation loss or auxiliary gradients
- Confidence scores not calibrated (always high or always low) — fix: [[Temperature Scaling]] or learned confidence head
- Right exits no better than left (information doesn't accumulate) — indicates bad lateral weight initialization or too-small `d_hidden`
- On FPGA: routing congestion kills clock speed — fix: reduce node fanout, add pipeline registers

---

### Recommended Next Steps (ordered)

1. **Python**: implement triangular version on MNIST/CIFAR-10, verify accuracy monotonicity
2. **Training**: try uniform loss first, then distillation, measure calibration
3. **FPGA sim**: write behavioral Verilog/VHDL of a 4×4 node, simulate in Verilator
4. **MCU**: port int8 forward pass to RP2040, time per-column latency

The core idea is sound and underexplored in the hardware-NN intersection. The calibration training is the hardest part to get right.



## Rectangular Grid + Right-Reads: Revised Design

### Why Rectangular is Better Here

Triangular was optimized for *one-shot anytime* — you were describing something richer: **iterative refinement via lateral message passing**. This is closer to [[Loopy Belief Propagation]] or a spatially-unrolled [[Recurrent Neural Network]].

```
input col
  ↓   ↓   ↓   ↓
→ N00→N01→N02→N03
  ↓   ↓   ↓   ↓
→ N10→N11→N12→N13
  ↓   ↓   ↓   ↓
→ N20→N21→N22→N23
  ↓   ↓   ↓   ↓
 out0 out1 out2 out3   ← caller aggregates/picks
```

Node reads: **top + left + right** (right = previous wave's right neighbor value)

---

### The Wave Mechanic

**Wave 1:** right-neighbor state = 0 (cold start). Left nodes compute fast, uninformed answers.

**Wave 2:** right-neighbor state = wave 1 output. Now every node gets lateral context from the right — far-right nodes have seen more of the input space and can correct leftward neighbors.

**Wave k:** nodes progressively converge. This is [[Message Passing]] on a grid, unrolled in time.

```python
class SystolicNode:
    def __init__(self, d):
        self.W_top  = np.random.randn(d, d) * 0.01
        self.W_left = np.random.randn(d, d) * 0.01
        self.W_right = np.random.randn(d, d) * 0.01  # reads prev-wave right neighbor
        self.b = np.zeros(d)
    
    def forward(self, h_top, h_left, h_right_prev):
        return np.tanh(
            self.W_top  @ h_top  +
            self.W_left @ h_left +
            self.W_right @ h_right_prev +  # recurrent via wave
            self.b
        )

class SystolicGrid:
    def __init__(self, rows, cols, d, n_classes):
        self.rows, self.cols, self.d = rows, cols, d
        self.nodes = {(r,c): SystolicNode(d) for r in range(rows) for c in range(cols)}
        self.heads = {c: np.random.randn(n_classes, d) * 0.01 for c in range(cols)}
        self.state = np.zeros((rows, cols, d))  # persists between waves
    
    def wave(self, x_cols):
        # x_cols: (cols, d) — one input vector per column, streamed left edge
        prev_state = self.state.copy()
        new_state = np.zeros_like(self.state)
        
        for r in range(self.rows):
            for c in range(self.cols):
                h_top   = x_cols[c] if r == 0 else new_state[r-1, c]
                h_left  = np.zeros(self.d) if c == 0 else new_state[r, c-1]
                h_right = np.zeros(self.d) if c == self.cols-1 else prev_state[r, c+1]
                new_state[r, c] = self.nodes[(r,c)].forward(h_top, h_left, h_right)
        
        self.state = new_state
        return self.read_bottom()
    
    def read_bottom(self):
        outputs = []
        for c in range(self.cols):
            h = self.state[self.rows-1, c]
            logits = self.heads[c] @ h
            probs = softmax(logits)
            conf = 1.0 - entropy(probs) / np.log(len(probs))
            outputs.append((probs, conf))
        return outputs
    
    def run(self, x_cols, max_waves=5, conf_thresh=0.9, 
            agg='confidence_weighted'):
        self.state[:] = 0  # reset
        for w in range(max_waves):
            outputs = self.wave(x_cols)
            result = self.aggregate(outputs, agg)
            max_conf = max(c for _, c in outputs)
            if max_conf >= conf_thresh:
                return result, w+1  # early exit
        return result, max_waves
    
    def aggregate(self, outputs, mode):
        probs_list = np.array([p for p,_ in outputs])
        confs = np.array([c for _,c in outputs])
        if mode == 'confidence_weighted':
            weights = confs / confs.sum()
            return probs_list.T @ weights
        elif mode == 'rightmost':
            return probs_list[-1]
        elif mode == 'mean':
            return probs_list.mean(axis=0)
        elif mode == 'pick_threshold':
            # caller picks leftmost node meeting threshold
            for i, (p, c) in enumerate(outputs):
                if c > 0.8:
                    return p
            return probs_list[-1]
```

---

### What This Actually Is

| Framing | Description |
|---|---|
| [[Cellular Automaton]] | Local rules, global behavior emerges |
| [[Iterative Refinement]] | Like diffusion models — coarse→fine |
| [[Bidirectional RNN]] | But spatial instead of temporal |
| [[Belief Propagation]] | Message passing to consensus |
| [[Hopfield Network]] | Energy-minimizing state over waves |

The right-read specifically makes this a **[[Graph Neural Network]]** with a grid topology, iterated k times. Convergence behavior follows from that literature.

---

### Caller API Design

Three natural control points:

```
1. Column selection  — pick col c for compute budget
2. Wave count        — more waves = more refined, until convergence
3. Aggregation mode  — mean / conf-weighted / threshold-first / rightmost
```

The caller can also **short-circuit mid-wave** if they only need the leftmost column output — the left columns complete first in a left→right sweep.

---

### FPGA Implications of Right-Read

The right dependency breaks single-wave parallelism. Two clean solutions:

**Ping-pong buffers** ← simpler
- Buffer A = wave N state, Buffer B = wave N+1 state being computed
- Right-reads come from Buffer A, writes go to Buffer B
- Flip on wave completion
- Full systolic parallelism within a wave still works on anti-diagonals

**Bidirectional pipeline** ← faster latency
- Left→right pass, then right→left pass per wave
- 2 clock cycles per row instead of 1
- Similar to how [[Elman Networks]] are unrolled

Both are FPGA-friendly. Ping-pong is the standard systolic trick.

---

### Failure Modes (updated)

| Failure | Cause | Fix |
|---|---|---|
| No convergence across waves | W_right too large, oscillation | Constrain spectral radius of W_right < 1 |
| Right-reads hurt not help | Wrong gradient flow during training | Train with fixed wave count, BPTT through waves |
| Left columns always ignored | Confidence never high enough early | Calibration loss or temperature per column |
| State explosion | State persists across inputs without reset | Explicit reset gate (GRU-style) per node |

---

### Training

Must **unroll waves** during backprop — treat k waves as k steps of a [[Recurrent Neural Network]]. Loss at every wave output:

```
L = Σ_w Σ_c λ_wc · CE(y, output[w][c])
```

Weight `λ` heavier on later waves and right columns to create the accuracy gradient. Otherwise wave 1 col 0 and wave 5 col 3 learn the same thing.

---

### Verdict on MCU

This version is *harder* on MCU — you need to store full grid state between waves (persistent SRAM). For a 4×4 grid, d=8: `4×4×8 floats = 512 bytes`, fine for RP2040. But you need two buffers for ping-pong: 1KB. Still fine. The wave loop fits naturally in a tight C loop and you get the anytime property by just reading after each wave.



### Halt Logic — the key design decision

Two independent stopping criteria, first-wins:

**1. State delta** — after each wave, compute `max(|state[wave] - state[wave-1]|)` at the bottom row. If `< DELTA_THRESH`, the grid has reached a [[Fixed Point]] — more waves are mathematically useless, not just probably useless. This is the cleaner signal.

**2. Confidence** — normalized entropy of the softmax output. `1.0` = completely certain, `0.0` = uniform. If `> CONF_THRESH`, exit early.

The caller controls `max_waves` as a hard budget. So the control flow is:

```
for each wave:
  run wave
  if delta < ε → done (converged)
  if conf > θ → done (certain)
  if wave == max_waves → done (budget)
```

The delta check is especially useful: if input is easy, the grid converges in 1-2 waves. Hard inputs take more. The network self-reports difficulty.

---

### Architecture of the C export

- **PCA projection** baked in as a static array — handles the `784→32` step on-device
- **Per-column input projections** `W_in_0..3` map the PCA output to `D`-vectors
- **Pointer tables** for weights instead of `switch` — compiler can vectorize loops
- **`systolic_predict(input, max_waves, out_probs, out_waves)`** — single call, returns class + fills probs + reports waves used (so caller knows how much compute was spent)
- Aggregation: **confidence-weighted average** across bottom-row columns — rightmost correct-but-slow nodes naturally dominate when they're confident

---

### To run

```bash
pip install scikit-learn numpy
python systolic_net.py
# → produces systolic_net.c
```

On RP2040, the PCA array (`784*32 floats = 100KB`) is too large for SRAM — put it in flash (`const` in C is enough on most ARM toolchains). Node weights at `D=16`: `4*4*3*(16*16) * 4 bytes ≈ 12KB`. Tight but fits.


### Dataset progression rationale

Each dataset isolates one specific architectural claim, in order of what can go wrong first:

**XOR/Parity** is the most important one and easy to overlook. It's the only test that specifically falsifies whether the lateral recurrent connection is doing *anything* — if column 0 and column 3 learn the same accuracy, the whole design is just a regular MLP in disguise. Run this first before spending any time on MNIST.

**MNIST → monotonicity check** is more diagnostic than raw accuracy. 96% is achievable by many broken architectures. What you want to see is the 4×4 accuracy grid (wave × column) where every cell is strictly better than the one to its left and above it. If it's not monotonic, training loss weighting needs fixing.

**Fashion-MNIST → halt correlation** tests whether the delta-based convergence is *meaningful* vs just a coincidental threshold. The key insight: if the network is self-aware about its uncertainty, *wrong* predictions should systematically use more waves than correct ones. If that correlation is weak or inverted, the halt criterion is noise.

**CIFAR-10** is deliberately hard — with PCA-64 input you won't get competitive absolute accuracy, and that's fine. You're measuring the shape of the compute-accuracy curve, not the ceiling. A Pareto-efficient curve (more waves → better, diminishing returns) is the pass condition.

**RP2040 budget**: the 5ms per wave target is conservative — 125MHz with float32 MACs and D=16 should be well under that. The constraint that matters is the PCA weight array in flash vs SRAM, which the C export already handles correctly with `static const`.


