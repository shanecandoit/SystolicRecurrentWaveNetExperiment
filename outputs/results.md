# SRWN Experiment Results

Generated: 2026-05-01 12:30:52

## What Was Run

Command:

```bash
D:/apps/Python39/python.exe srwn_experiment.py --json
```

Configuration:

| Parameter | Value |
|---|---|
| seed | 7 |
| rows | 3 |
| cols | 4 |
| hidden_dim | 12 |
| train_waves | 4 |
| eval_waves | 4 |
| learning_rate | 0.01 |
| epochs | 120 |
| batch_size | 128 |
| train_size | 4096 |
| val_size | 1024 |
| test_size | 2048 |
| conf_threshold | 0.9 |

## Recurrent Accuracy Grid

| Wave | Col 0 | Col 1 | Col 2 | Col 3 |
|---|---|---|---|---|
| 1 | 0.5186 | 0.4995 | 0.5107 | 0.9932 |
| 2 | 0.4995 | 0.5073 | 1.0000 | 1.0000 |
| 3 | 0.4985 | 1.0000 | 1.0000 | 0.9932 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Ablated Accuracy Grid

| Wave | Col 0 | Col 1 | Col 2 | Col 3 |
|---|---|---|---|---|
| 1 | 0.5186 | 0.5137 | 0.5049 | 0.9429 |
| 2 | 0.5186 | 0.5137 | 0.5049 | 0.9429 |
| 3 | 0.5186 | 0.5137 | 0.5049 | 0.9429 |
| 4 | 0.5186 | 0.5137 | 0.5049 | 0.9429 |

## Key Metrics

| Metric | Recurrent | Ablated |
|---|---|---|
| Horizontal monotonicity | 0.8333 | 0.3333 |
| Vertical monotonicity | 0.7500 | 1.0000 |
| Halt mean wave | 1.2593 | 1.7236 |
| Halted-policy accuracy | 1.0000 | 0.9473 |

## Expected vs Observed

| Check | Expected | Observed | Status |
|---|---|---|---|
| recurrent improves early columns | large early-column gains with recurrence | col0 gain=0.4814, col1 gain=0.5005 | PASS |
| ablated stays flat | little or no early-column gain without recurrence | col0 gain=0.0000, col1 gain=0.0000 | PASS |
| rightmost final accuracy | rightmost final accuracy above 0.95 | 1.0000 | PASS |
| delta convergence signal | delta should shrink across later waves | wave2=2.0000, wave4=1.9981 | FAIL |

## How To Interpret

1. If recurrent early-column gains are high while ablated gains are near zero, the recurrent wave path is contributing meaningful refinement.
2. If rightmost final accuracy is high but delta convergence fails, classification works but fixed-point convergence is not yet validated.
3. Halted-policy accuracy close to rightmost final accuracy means confidence-threshold exit is viable on this task.
4. For direct comparison against a conventional MLP baseline, read outputs/benchmark_results.md.

## Evaluation

1. Recurrent left-column gain from wave 1 to wave 4: col0=0.4814, col1=0.5005.
2. Ablated left-column gain from wave 1 to wave 4: col0=0.0000, col1=0.0000.
3. Recurrent refinement appears active if recurrent gains exceed ablated gains on early columns.
4. Delta-based convergence remains questionable when recurrent delta stays high across late waves.

## New Baseline Comparison (80-epoch run)

Source: outputs/benchmark_metrics.json and outputs/benchmark_results.md.

| Model | Test accuracy | Estimated MACs/sample | Parameters |
|---|---:|---:|---:|
| SRWN recurrent fixed | 1.0000 | 21216.00 | 5576 |
| SRWN recurrent adaptive | 1.0000 | 6745.10 | 5576 |
| SRWN ablated fixed | 0.9951 | 21216.00 | 5576 |
| MLP baseline | 0.9961 | 816.00 | 866 |

Interpretation:

1. Accuracy: recurrent SRWN is best on this parity task (1.0000), slightly above MLP (0.9961).
2. Compute: even adaptive SRWN remains much heavier than MLP in this configuration (6745 vs 816 MACs/sample).
3. Recurrence value: recurrent SRWN beats ablated SRWN in final test accuracy (1.0000 vs 0.9951), indicating measurable benefit from the recurrent path.
4. Anytime value: adaptive halting cuts recurrent SRWN compute by about 68 percent vs fixed-wave SRWN (21216 -> 6745 MACs/sample) while preserving accuracy.

## Plot Summaries

### 1. outputs/train_loss.png

- SRWN recurrent (blue) and SRWN ablated (orange) stay near chance-level loss (~0.69) until around epochs 30-35, then diverge.
- SRWN recurrent drops sharply and stabilizes around ~0.17, indicating a clean fit.
- SRWN ablated improves partially but plateaus around ~0.46, suggesting optimization stalls without recurrent feedback.
- MLP (green) learns later (around epochs 54-56) and then drops to very low loss (~0.02), showing strong eventual fit despite slower onset.

### 2. outputs/val_accuracy.png

- All models hover near ~0.5 early (random-guess regime).
- SRWN ablated jumps first (around epoch ~30), recurrent SRWN follows shortly after and reaches 1.0.
- MLP improves later than both SRWN variants, then climbs to near-perfect validation accuracy by the end.
- Temporary dips in SRWN curves suggest training instability around the transition point, but final validation performance is near-saturated for all models.

### 3. outputs/accuracy_vs_compute.png

- The plot separates models primarily by compute, not by accuracy, because all points are near 0.995-1.000 test accuracy.
- MLP is far left (lowest compute) with slightly lower accuracy than recurrent SRWN.
- SRWN recurrent fixed is far right (highest compute) with top accuracy.
- SRWN recurrent adaptive shifts left substantially vs fixed SRWN while keeping top accuracy, demonstrating internal compute reduction.
- SRWN ablated fixed has similar compute to recurrent fixed but lower accuracy, so recurrence improves quality at essentially equal fixed compute.

## Overall Conclusion

On parity, SRWN achieves best accuracy and recurrence clearly helps refinement, but compute efficiency against a conventional MLP is not yet competitive under current dimensions. The strongest short-term direction is to shrink SRWN cost (hidden size, rows/cols, and wave budget) and rerun the same plots to look for a better compute-accuracy tradeoff.

## Fashion-MNIST Experiment (SRWN vs CNN)

### What was run

Command:

```bash
D:/apps/Python39/python.exe fashion_mnist_benchmark.py --out-dir outputs
```

Configuration highlights:

- train/val/test: 12000 / 2000 / 5000
- epochs: 6
- SRWN: rows=3, cols=4, hidden_dim=24, train_waves=3, eval_waves=3
- baseline: 2-layer CNN

### Final results

Source: outputs/fashion_metrics.json and outputs/fashion_results.md.

| Model | Test accuracy | Estimated MACs/sample | Parameters |
|---|---:|---:|---:|
| SRWN fixed | 0.8298 | 83904.00 | 40936 |
| SRWN adaptive | 0.8212 | 72678.49 | 40936 |
| CNN baseline | 0.8706 | 1218048.00 | 206922 |

### Findings

1. CNN is more accurate on this run: 0.8706 vs SRWN fixed 0.8298.
2. SRWN is substantially cheaper in estimated MACs than CNN (about 14.5x lower for SRWN fixed and about 16.8x lower for SRWN adaptive).
3. SRWN adaptive mode reduces compute relative to SRWN fixed by about 13.4 percent (83904 -> 72678 MACs/sample), with a modest accuracy drop (0.8298 -> 0.8212).
4. Compared with the parity benchmark, Fashion-MNIST exposes a clearer tradeoff: SRWN currently wins on compute, CNN wins on accuracy.

### Plot summaries

#### 1. outputs/fashion_train_loss.png

- Both models improve steadily over all 6 epochs.
- CNN keeps lower training loss than SRWN throughout.
- SRWN drops from ~1.55 to ~0.48, which indicates learning is working but still underfits relative to CNN at this epoch budget.

#### 2. outputs/fashion_val_accuracy.png

- Both curves rise consistently, so optimization is stable.
- CNN stays above SRWN at every epoch and ends around 0.8805 validation accuracy.
- SRWN climbs from 0.7340 to about 0.8405, suggesting more training or a stronger SRWN configuration may still improve performance.

#### 3. outputs/fashion_accuracy_vs_compute.png

- CNN sits in the high-accuracy, high-compute corner.
- SRWN fixed and adaptive sit in lower-compute but lower-accuracy positions.
- The plot demonstrates a realistic efficiency frontier: SRWN is currently a compute-saving option, while CNN is the accuracy-leading option.

### Bottom line for Fashion-MNIST

On this setup, SRWN does not beat CNN in accuracy yet, but it achieves much lower compute cost. The immediate next step is to tune SRWN (hidden size, waves, and column layout) to recover part of the accuracy gap while staying below CNN compute.

## Fashion-MNIST v2 (deeper and wider SRWN sweep)

### What was run

Command:

```bash
D:/apps/Python39/python.exe fashion-mnist-v2.py --out-dir outputs --epochs 8
```

Artifacts:

- outputs/fashion_v2_metrics.json
- outputs/fashion_v2_verdict.json
- outputs/fashion_v2_results.md
- outputs/fashion_v2_accuracy_bar.png
- outputs/fashion_v2_compute_bar.png
- outputs/fashion_v2_acc_vs_macs.png

### SRWN variants vs CNN baseline

| Variant | Family | Rows | Hidden dim | Eval waves | Test acc | Adaptive MACs/sample |
|---|---|---:|---:|---:|---:|---:|
| baseline_r3_h24_w3 | baseline | 3 | 24 | 3 | 0.8328 | 65826.89 |
| deeper_r4_h24_w4 | deeper | 4 | 24 | 4 | 0.8366 | 97573.82 |
| deeper_r5_h24_w4 | deeper | 5 | 24 | 4 | 0.8256 | 131940.10 |
| wider_r3_h32_w3 | wider | 3 | 32 | 3 | 0.8424 | 103138.25 |
| wider_r3_h48_w3 | wider | 3 | 48 | 3 | 0.8532 | 202217.24 |
| cnn_baseline | cnn | - | - | - | 0.8700 | 1218048.00 |

### When do we beat CNN on both accuracy and MACs?

Short answer from this v2 sweep: no tested SRWN variant beat CNN on both metrics simultaneously.

- Best SRWN accuracy: wider_r3_h48_w3 at 0.8532 (still below CNN at 0.8700)
- All SRWN variants remained far below CNN compute (about 6x to 18x cheaper)

### Plot summaries (v2)

#### 1. outputs/fashion_v2_accuracy_bar.png

- Accuracy generally improved when SRWN was widened.
- The best bar among SRWN variants was wider_r3_h48_w3.
- CNN baseline remained the highest-accuracy model in this run.

#### 2. outputs/fashion_v2_compute_bar.png

- CNN has by far the largest compute budget.
- Wider and deeper SRWN variants increase compute as expected.
- Even the heaviest SRWN variant stayed well below CNN MACs.

#### 3. outputs/fashion_v2_acc_vs_macs.png

- SRWN points form a lower-compute, lower-accuracy cluster.
- CNN sits at higher accuracy but much higher compute.
- No SRWN point crosses above CNN in accuracy while staying below it in compute.

### v2 conclusion

Deeper and wider SRWN variants improved Fashion-MNIST accuracy, especially wider models, but none closed the full gap to CNN. The current frontier is still: SRWN for lower compute, CNN for higher accuracy.

## Fashion-MNIST SRWN Hyperparameter Sweep (small)

### Sweep setup

Command pattern used (4 trials):

```bash
D:/apps/Python39/python.exe fashion_mnist_benchmark.py --out-dir outputs/fashion_sweep/<config> --epochs 4 --train-size 8000 --val-size 1500 --test-size 3000
```

Sweep summary file:

- outputs/fashion_sweep/sweep_summary.json

### Best-config table

| Config | Rows | Cols | Hidden dim | Eval waves | Rightmost test acc | Halted test acc | Adaptive MACs/sample | Params | CNN acc (same run) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cfg1_r3c4h16w2 | 3 | 4 | 16 | 2 | 0.7557 | 0.7487 | 32256.00 | 22696 | 0.8433 |
| cfg2_r3c4h24w2 | 3 | 4 | 24 | 2 | 0.7947 | 0.7810 | 62208.00 | 40936 | 0.8367 |
| cfg3_r3c4h24w3 | 3 | 4 | 24 | 3 | 0.7840 | 0.7803 | 83904.00 | 40936 | 0.8423 |
| cfg4_r4c4h24w3 | 4 | 4 | 24 | 3 | 0.7917 | 0.7810 | 109920.00 | 53512 | 0.8440 |

### Sweep findings

1. Best accuracy config in this small sweep: cfg2_r3c4h24w2 (rightmost test accuracy 0.7947).
2. Best accuracy-per-compute config: cfg1_r3c4h16w2 (lower accuracy but much lower compute).
3. Increasing waves from 2 to 3 at hidden_dim=24 did not improve accuracy here (cfg2 > cfg3) and increased compute substantially.
4. Increasing rows from 3 to 4 at hidden_dim=24 and waves=3 did not beat the best-accuracy config, while further increasing compute.
5. In all four trials, mean halt wave hit the eval-wave ceiling, so adaptive halting did not reduce compute inside these sweep settings.

### Practical selection guidance

- If priority is highest SRWN accuracy from this sweep, choose cfg2_r3c4h24w2.
- If priority is SRWN efficiency, choose cfg1_r3c4h16w2.
- For next sweep, keep waves at 2 and vary hidden_dim and learning rate before increasing rows or waves.
