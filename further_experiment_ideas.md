# Further Experiment Ideas

Based on what the current experiments established and where they fell short.

## What the current experiments showed

- **Recurrence is meaningful.** On parity, the recurrent wave path improves early-column accuracy across waves; the ablated model stays flat. This is the core architectural claim and it holds.
- **Adaptive halting works.** Confidence-threshold early exit reduces MACs significantly without a large accuracy drop (parity: comparable accuracy; Fashion-MNIST: ~14% MAC reduction).
- **SRWN is compute-cheap but accuracy-limited on Fashion-MNIST.** Fixed SRWN (83,904 MACs/sample) and adaptive SRWN (72,678 MACs/sample) are over 14× cheaper than the CNN baseline (1,218,048 MACs/sample), but trail it on accuracy (0.83 vs 0.87).
- **Delta-convergence is not validated.** The hidden-state delta between waves does not reliably shrink — confidence-threshold halting is more useful than delta-threshold halting under the current setup.

---

## Idea 1: Close the accuracy gap on Fashion-MNIST

The most direct open question. SRWN is 4+ points behind CNN on a task where it has a massive compute advantage. Closing that gap would make the tradeoff compelling.

**Subexperiments:**
- **Deeper grid** — increase `rows` from 3 to 5 or 6. More rows means more sequential processing depth within each wave at the cost of more MACs per wave.
- **Wider hidden dimension** — try `hidden_dim` 48 or 64 (current: 24). GridCells are the bottleneck representational unit.
- **More waves** — train with 5–6 waves instead of 3. More recurrent refinement passes may help with a harder 10-class task.
- **Patch-based input rather than horizontal strips** — currently the 28×28 image is split into `cols` horizontal strips (196 pixels each). Splitting into 2D patches (e.g. 4 non-overlapping 14×14 patches) may give each column a more spatially coherent local view, which is a better inductive bias for image data.
- **Convolutional input projection** — replace each column's `nn.Linear(chunk_dim, hidden_dim)` with a small conv + pool over its spatial chunk. This gives the grid a better front-end without changing the wave architecture.

---

## Idea 2: Validate or replace delta-convergence halting

The current experiments show delta does not reliably shrink across waves, so the delta-based convergence criterion fails. Two directions:

- **Supervised convergence signal** — add a small binary "done" head to each cell trained to predict whether adding another wave will change the output. This is similar to the ACT (Adaptive Computation Time) approach but at the cell level.
- **Learned halting scalar** — replace the entropy-based confidence with a learned scalar per (wave, col) trained with a Halt/Continue label derived from whether the final-wave prediction would change. This is more directly optimizable than an entropy heuristic.
- **Per-sample delta tracking** — rather than looking at mean delta across the batch, track per-sample delta and compare it to per-sample confidence. Check whether samples that halt early (by confidence) also have low delta — if so, delta and confidence are correlated and either would work; if not, they capture different convergence signals.

---

## Idea 3: Longer parity sequences

Current parity uses 8-bit sequences across 4 columns (2 bits per column). Testing at 16 or 32 bits with proportionally more columns would stress both the horizontal propagation (left/right communication across columns) and the wave depth needed to integrate global information.

This is a clean controllable environment — unlike image benchmarks, you know exactly what information must flow where. If SRWN recurrence fails to scale with sequence length, that is a hard architectural limit. If it scales, it suggests the grid can route long-range dependencies.

**Metric:** plot accuracy vs bit count for recurrent vs ablated SRWN. The gap between the two curves is the net value of recurrence as a function of task difficulty.

---

## Idea 4: Compute-accuracy Pareto frontier

The current comparison is a single operating point per model. A sweep over configurations would show the shape of the tradeoff curve:

- For SRWN: sweep `(rows, cols, hidden_dim, eval_waves)` at fixed training budget.
- For CNN: sweep `(conv channels, fc width)`.
- Plot test accuracy vs MACs for every configuration on the same axes.

If SRWN occupies the low-MAC end of the Pareto frontier and CNN occupies the high-accuracy end, the picture is one of architectural specialization — SRWN is a legitimate option for compute-constrained deployment, not a strictly worse model.

---

## Idea 5: Bidirectional or alternating wave direction

Currently waves always sweep the same direction: top-to-bottom within each wave, left-to-right recurrence from the previous wave's right neighbor. This means information from the right side of the input takes multiple waves to reach the left side.

- **Alternating sweep direction** — odd waves use the current left-to-right recurrence; even waves flip and use right-to-left recurrence (reading from `col - 1` of the previous state instead of `col + 1`). This halves the number of waves needed for full bidirectional coverage.
- **Bidirectional within each wave** — run two passes per wave (one left-to-right, one right-to-left) and sum or concatenate their outputs before updating state. More expensive per wave but potentially faster convergence.

---

## Idea 6: Monotonicity regularization

The current code measures horizontal and vertical monotonicity of accuracy across the grid but does not enforce it. On parity the recurrent model achieves good monotonicity naturally, but on harder tasks this may not hold.

Add a soft penalty during training: if the logit confidence at `(wave w, col c)` is lower than at `(wave w-1, col c)` or `(wave w, col c-1)`, penalize the difference. This encourages the model to treat additional waves and later columns as refinements rather than independent predictions.

---

## Idea 7: Comparison to early-exit baselines

SRWN's anytime behavior (produce a useful prediction at any wave, improve with more computation) is also the goal of early-exit networks (e.g. BranchyNet, Multi-Scale DenseNets). Running a fair comparison on Fashion-MNIST would contextualize where SRWN sits:

- **Early-exit MLP** — a 3-layer MLP where each layer also has a classification head; exit when confidence exceeds threshold.
- **Early-exit CNN** — a CNN where intermediate feature maps feed classification heads, with the same confidence threshold used for SRWN.

The question is whether SRWN's grid recurrence structure gives it better accuracy-per-MAC at low-compute operating points than a structurally simpler early-exit model.

---

## Idea 8: Sequence tasks (time-series or simple NLP)

Parity and Fashion-MNIST test SRWN on parallel input (all columns available simultaneously). A sequential task — e.g. character-level prediction on short strings, or anomaly detection in a time series — would test whether the column-to-column communication can model temporal dependencies, not just spatial ones.

If the column index maps to time position (earlier columns = earlier timesteps), the left-to-right wave propagation is a natural match for causal inference, with additional waves providing a kind of global context refinement.

---

## Priority ordering (suggested)

1. **Idea 1 (accuracy gap)** — highest impact; determines whether SRWN is practically useful on real image tasks.
2. **Idea 4 (Pareto sweep)** — clarifies the story without new architecture changes, just more runs.
3. **Idea 3 (longer parity)** — clean probe of the core recurrence claim under harder conditions.
4. **Idea 5 (bidirectional waves)** — low implementation cost, potentially large accuracy benefit.
5. **Ideas 2, 6, 7, 8** — more involved; better pursued once the accuracy gap is better understood.
