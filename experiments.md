# SRWN Experiments

## Objective

Validate the first architectural claim before moving to larger datasets:

> Recurrent right-neighbor reads should let early columns improve across waves when the input is distributed across columns.

## Experiment Design

### Task

8-bit parity classification.

- input: 8 binary features
- label: parity of all 8 bits
- columns: 4
- locality: each column receives only its own 2-bit slice

This locality constraint matters. Without it, any column could solve the task directly from the full input and the experiment would fail to isolate the recurrent mechanism.

### Models Compared

1. `recurrent`: standard SRWN with right-neighbor previous-wave reads enabled
2. `ablated`: identical SRWN with the recurrent right-read path removed

### Metrics

1. Accuracy for every `(wave, column)` pair
2. Horizontal monotonicity ratio: fraction of adjacent columns where accuracy does not decrease
3. Vertical monotonicity ratio: fraction of adjacent waves where accuracy does not decrease
4. Confidence-halting summary using the earliest wave where any column exceeds the confidence threshold

### Pass Condition

The recurrent model should materially improve left and middle columns over later waves, while the ablated model should not.

## Run Command

```bash
D:/apps/Python39/python.exe srwn_experiment.py --json
```

## Results

Command used:

```bash
D:/apps/Python39/python.exe srwn_experiment.py --json
```

Configuration used:

| Parameter | Value |
|---|---|
| Rows | 3 |
| Cols | 4 |
| Hidden dim | 12 |
| Train waves | 4 |
| Epochs | 120 |
| Train / val / test | 4096 / 1024 / 2048 |
| Confidence threshold | 0.9 |

Measured test accuracy grid, recurrent model:

| Wave | Col 0 | Col 1 | Col 2 | Col 3 |
|---|---|---|---|---|
| 1 | 0.5186 | 0.4995 | 0.5107 | 0.9932 |
| 2 | 0.4995 | 0.5073 | 1.0000 | 1.0000 |
| 3 | 0.4985 | 1.0000 | 1.0000 | 0.9932 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Measured test accuracy grid, ablated model:

| Wave | Col 0 | Col 1 | Col 2 | Col 3 |
|---|---|---|---|---|
| 1 | 0.5186 | 0.5137 | 0.5049 | 0.9429 |
| 2 | 0.5186 | 0.5137 | 0.5049 | 0.9429 |
| 3 | 0.5186 | 0.5137 | 0.5049 | 0.9429 |
| 4 | 0.5186 | 0.5137 | 0.5049 | 0.9429 |

Auxiliary metrics:

| Metric | Recurrent | Ablated |
|---|---|---|
| Horizontal monotonicity | 0.8333 | 0.3333 |
| Vertical monotonicity | 0.7500 | 1.0000 |
| Halt mean wave | 1.2593 | 1.7236 |
| Halted-policy accuracy | 1.0000 | 0.9473 |
| Threshold hit rate | 0.9932 | 0.7588 |
| Mean bottom-row delta at wave 2 | 2.0000 | 0.0000 |
| Mean bottom-row delta at wave 3 | 2.0000 | 0.0000 |
| Mean bottom-row delta at wave 4 | 1.9981 | 0.0000 |

## Findings

1. The recurrent path is doing real work. In the ablated model, every wave is identical. In the recurrent model, columns 1 and 2 become perfect only after later waves, and column 0 reaches perfect accuracy by wave 4.
2. The core parity claim is validated for early exits. Left and middle columns improve across waves only when right-neighbor recurrence is present.
3. The rightmost column already solves the task almost immediately. Column 3 reaches 99.32 percent accuracy at wave 1 even in the recurrent model, so this benchmark mainly validates backward refinement into earlier columns rather than a strong need for multi-wave final prediction quality.
4. Confidence thresholding looks usable on this toy benchmark. The recurrent model halts at a mean of 1.26 waves with 100 percent halted-policy accuracy.
5. Delta-based convergence is not validated. The recurrent model's bottom-row delta stays near 2.0 across later waves instead of decaying, so the current design does not behave like a convergent fixed-point process on this run.

## Flaws Exposed By The Experiment

1. The current benchmark is strong for validating recurrent refinement into early columns, but weak for validating temporal anytime on the selected output policy because the rightmost column is already nearly perfect after one wave.
2. The convergence story in the design is currently unsupported. If wave-to-wave delta stays large while accuracy is already perfect, then delta-threshold halting needs additional constraints, different activations, or explicit recurrent stabilization.
3. Confidence success on this task should not be overgeneralized. The parity benchmark is too clean to establish calibration quality on realistic data.

## Next Step

The next useful experiment is MNIST with locality-preserving slices or masked projections, because it can test whether recurrence still improves early exits when the rightmost column does not trivially solve the task in one wave.

## Interpretation Template

If the recurrent model improves early columns and the ablated model does not, the core SRWN recurrence claim survives first contact.

If both models behave similarly, then either:

- the recurrent path is not contributing meaningful information
- the training objective is not incentivizing early exits correctly
- the data presentation still lets columns bypass the intended locality constraint

If confidence crosses threshold early for both correct and incorrect samples at similar rates, halting is not validated and needs separate calibration work.