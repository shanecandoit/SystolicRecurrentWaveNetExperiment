# SRWN Design

## Goal

SRWN is intended to provide two kinds of adaptive compute:

- spatial anytime: earlier columns are cheaper and later columns are richer
- temporal anytime: repeated waves refine the same grid state over time

The intended value proposition is not peak accuracy alone. The design is useful only if it improves the compute-accuracy curve by producing usable early exits and refining them when more budget is available.

## Core Architecture

The model is a rectangular `ROWS x COLS` grid of hidden states with hidden size `D`.

For each node `N[r,c]` at a given wave:

```text
z[r,c] = W_top[r,c]   * h[r-1, c]      current wave
       + W_left[r,c]  * h[r, c-1]      current wave
       + W_right[r,c] * h_prev[r, c+1] previous wave
       + b[r,c]

h[r,c] = tanh(z[r,c])
```

Boundary conditions:

- top row reads projected input for that column
- left edge reads zeros
- right edge reads zeros
- wave 0 uses zero previous state everywhere

Each bottom-row state `h[ROWS-1, c]` feeds a classifier head for column `c`.

## Architectural Hypotheses

1. Rightmost columns should outperform leftmost columns because they accumulate more same-wave context.
2. Earlier columns should improve across waves only when recurrent right-reads are active.
3. Confidence and convergence signals may be usable as self-halting criteria, but that is secondary until the recurrent improvement claim is validated.

## Known Design Risks

### Input locality is easy to violate

If every column can directly view the full input, then parity-style validation does not prove anything about lateral communication. The experiment harness therefore assigns each column a local slice of the input so left columns actually need recurrent feedback to improve.

### Raw accuracy can hide architectural failure

A rightmost output may still become strong even if early exits are useless. Validation must therefore inspect the full wave-by-column accuracy grid, not just the final aggregated prediction.

### Halting may be poorly calibrated

Confidence-based early exit is only useful if high confidence correlates with correctness. This needs dedicated evaluation and should not be assumed from classification accuracy.

## Validation Criteria

Phase 1 validates recurrent communication on synthetic parity.

Success criteria:

1. With recurrence enabled, left and middle columns improve across waves.
2. Without recurrence, repeated waves provide little or no benefit to left columns.
3. Final-wave rightmost accuracy remains high enough to show the model can solve the task.

Failure criteria:

1. Left columns do not improve with recurrence.
2. The recurrent and ablated variants look materially the same.
3. Monotonicity is weak enough that the anytime interpretation is not credible.

## Recommended Next Validation Steps

1. Repeat the same ablation on MNIST with locality-preserving input slicing.
2. Test whether confidence thresholding reduces mean compute at fixed accuracy.
3. Measure whether delta-based convergence is stable under quantization and C export.
