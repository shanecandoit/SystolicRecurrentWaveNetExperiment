import argparse
import copy
import json
import os
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F


TOTAL_BITS = 8
N_CLASSES = 2


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_parity_dataset(count: int, cols: int, seed: int):
    if TOTAL_BITS % cols != 0:
        raise ValueError("TOTAL_BITS must be divisible by cols for local input slicing")

    generator = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (count, TOTAL_BITS), generator=generator).float()
    labels = bits.sum(dim=1).long() % 2
    bits_per_col = TOTAL_BITS // cols
    column_inputs = bits.view(count, cols, bits_per_col)
    return column_inputs, labels


class GridCell(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.top = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.left = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.right = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, h_top, h_left, h_right):
        z = self.top(h_top) + self.left(h_left) + self.right(h_right) + self.bias
        return torch.tanh(z)


class SRWN(nn.Module):
    def __init__(self, rows: int, cols: int, hidden_dim: int, recurrent: bool):
        super().__init__()
        if TOTAL_BITS % cols != 0:
            raise ValueError("TOTAL_BITS must be divisible by cols")

        self.rows = rows
        self.cols = cols
        self.hidden_dim = hidden_dim
        self.recurrent = recurrent
        bits_per_col = TOTAL_BITS // cols

        self.input_layers = nn.ModuleList(
            [nn.Linear(bits_per_col, hidden_dim) for _ in range(cols)]
        )
        self.cells = nn.ModuleList(
            [GridCell(hidden_dim) for _ in range(rows * cols)]
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, N_CLASSES) for _ in range(cols)]
        )

    def cell(self, row: int, col: int) -> GridCell:
        return self.cells[row * self.cols + col]

    def forward(self, column_inputs, waves: int):
        batch_size = column_inputs.shape[0]
        device = column_inputs.device
        zero = torch.zeros(batch_size, self.hidden_dim, device=device)

        projected = torch.stack(
            [self.input_layers[col](column_inputs[:, col, :]) for col in range(self.cols)],
            dim=1,
        )

        state = torch.zeros(batch_size, self.rows, self.cols, self.hidden_dim, device=device)
        outputs = []
        confidences = []
        deltas = []

        for wave in range(waves):
            prev_state = state
            next_rows = []

            for row in range(self.rows):
                next_cols = []
                for col in range(self.cols):
                    h_top = projected[:, col, :] if row == 0 else next_rows[row - 1][:, col, :]
                    h_left = zero if col == 0 else next_cols[col - 1]
                    if col == self.cols - 1 or not self.recurrent:
                        h_right = zero
                    else:
                        h_right = prev_state[:, row, col + 1, :]

                    next_cols.append(self.cell(row, col)(h_top, h_left, h_right))

                next_rows.append(torch.stack(next_cols, dim=1))

            state = torch.stack(next_rows, dim=1)
            bottom = state[:, self.rows - 1, :, :]
            logits = torch.stack([self.heads[col](bottom[:, col, :]) for col in range(self.cols)], dim=1)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)
            confidence = 1.0 - entropy / torch.log(torch.tensor(float(N_CLASSES), device=device))
            delta = (
                torch.full((batch_size,), float("inf"), device=device)
                if wave == 0
                else (state[:, self.rows - 1] - prev_state[:, self.rows - 1]).abs().amax(dim=(1, 2))
            )

            outputs.append(logits)
            confidences.append(confidence)
            deltas.append(delta)

        return torch.stack(outputs, dim=1), torch.stack(confidences, dim=1), torch.stack(deltas, dim=1)


def accuracy_grid(logits, labels):
    predictions = logits.argmax(dim=-1)
    return (predictions == labels[:, None, None]).float().mean(dim=0)


def monotonicity_summary(acc_grid):
    horizontal = []
    for wave in range(len(acc_grid)):
        for col in range(len(acc_grid[wave]) - 1):
            horizontal.append(acc_grid[wave][col + 1] >= acc_grid[wave][col])

    vertical = []
    for wave in range(len(acc_grid) - 1):
        for col in range(len(acc_grid[wave])):
            vertical.append(acc_grid[wave + 1][col] >= acc_grid[wave][col])

    return {
        "horizontal": round(sum(horizontal) / len(horizontal), 4),
        "vertical": round(sum(vertical) / len(vertical), 4),
    }


def halting_summary(logits, confidences, labels, conf_threshold: float):
    batch_size = labels.shape[0]
    final_predictions = logits[:, -1, -1, :].argmax(dim=-1)
    correct = final_predictions == labels
    best_conf = confidences.max(dim=2).values
    best_col = confidences.argmax(dim=2)
    reached = best_conf >= conf_threshold
    first_wave = reached.float().argmax(dim=1) + 1
    first_wave = torch.where(reached.any(dim=1), first_wave, torch.full_like(first_wave, best_conf.shape[1]))
    wave_index = first_wave - 1
    selected_col = best_col[torch.arange(batch_size), wave_index]
    selected_logits = logits[torch.arange(batch_size), wave_index, selected_col, :]
    halted_predictions = selected_logits.argmax(dim=-1)

    return {
        "mean_wave": round(first_wave.float().mean().item(), 4),
        "mean_wave_correct": round(first_wave[correct].float().mean().item(), 4),
        "mean_wave_incorrect": round(first_wave[~correct].float().mean().item(), 4) if (~correct).any() else None,
        "threshold_hit_rate": round(reached.any(dim=1).float().mean().item(), 4),
        "halted_accuracy": round((halted_predictions == labels).float().mean().item(), 4),
    }


def evaluate(model, features, labels, waves: int, conf_threshold: float):
    model.eval()
    with torch.no_grad():
        logits, confidences, deltas = model(features, waves)
        acc_tensor = accuracy_grid(logits, labels)
        acc_grid = [[round(value, 4) for value in row] for row in acc_tensor.tolist()]
        return {
            "accuracy_grid": acc_grid,
            "monotonicity": monotonicity_summary(acc_grid),
            "halting": halting_summary(logits, confidences, labels, conf_threshold),
            "delta_mean_by_wave": [None] + [round(value, 4) for value in deltas.mean(dim=0).tolist()[1:]],
        }


def train_model(config, recurrent: bool):
    set_seed(config["seed"])
    model = SRWN(
        rows=config["rows"],
        cols=config["cols"],
        hidden_dim=config["hidden_dim"],
        recurrent=recurrent,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    train_x, train_y = build_parity_dataset(config["train_size"], config["cols"], config["seed"])
    val_x, val_y = build_parity_dataset(config["val_size"], config["cols"], config["seed"] + 1)

    wave_weights = torch.linspace(0.5, 1.0, config["train_waves"])
    col_weights = torch.linspace(0.5, 1.0, config["cols"])
    total_weight = float((wave_weights[:, None] * col_weights[None, :]).sum().item())

    best_state = None
    best_val = -1.0

    for _ in range(config["epochs"]):
        permutation = torch.randperm(train_x.shape[0])
        model.train()
        for start in range(0, train_x.shape[0], config["batch_size"]):
            idx = permutation[start : start + config["batch_size"]]
            batch_x = train_x[idx]
            batch_y = train_y[idx]

            optimizer.zero_grad()
            logits, _, _ = model(batch_x, config["train_waves"])
            loss = 0.0
            for wave in range(config["train_waves"]):
                for col in range(config["cols"]):
                    weight = wave_weights[wave] * col_weights[col]
                    loss = loss + weight * F.cross_entropy(logits[:, wave, col, :], batch_y)
            (loss / total_weight).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _, _ = model(val_x, config["eval_waves"])
            val_score = (val_logits[:, -1, -1, :].argmax(dim=-1) == val_y).float().mean().item()
            if val_score > best_val:
                best_val = val_score
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model


def run_experiment():
    config = {
        "seed": 7,
        "rows": 3,
        "cols": 4,
        "hidden_dim": 12,
        "train_waves": 4,
        "eval_waves": 4,
        "learning_rate": 0.01,
        "epochs": 120,
        "batch_size": 128,
        "train_size": 4096,
        "val_size": 1024,
        "test_size": 2048,
        "conf_threshold": 0.9,
    }

    test_x, test_y = build_parity_dataset(config["test_size"], config["cols"], config["seed"] + 2)
    recurrent_model = train_model(config, recurrent=True)
    ablated_model = train_model(config, recurrent=False)

    recurrent_metrics = evaluate(
        recurrent_model,
        test_x,
        test_y,
        waves=config["eval_waves"],
        conf_threshold=config["conf_threshold"],
    )
    ablated_metrics = evaluate(
        ablated_model,
        test_x,
        test_y,
        waves=config["eval_waves"],
        conf_threshold=config["conf_threshold"],
    )

    return {
        "config": config,
        "recurrent": recurrent_metrics,
        "ablated": ablated_metrics,
    }


def grid_to_markdown(grid):
    header = "| Wave | Col 0 | Col 1 | Col 2 | Col 3 |"
    sep = "|---|---|---|---|---|"
    rows = [header, sep]
    for idx, row in enumerate(grid, start=1):
        rows.append(f"| {idx} | {row[0]:.4f} | {row[1]:.4f} | {row[2]:.4f} | {row[3]:.4f} |")
    return "\n".join(rows)


def evaluate_findings(results):
    rec_grid = results["recurrent"]["accuracy_grid"]
    abl_grid = results["ablated"]["accuracy_grid"]

    rec_col0_gain = round(rec_grid[-1][0] - rec_grid[0][0], 4)
    abl_col0_gain = round(abl_grid[-1][0] - abl_grid[0][0], 4)
    rec_col1_gain = round(rec_grid[-1][1] - rec_grid[0][1], 4)
    abl_col1_gain = round(abl_grid[-1][1] - abl_grid[0][1], 4)

    findings = [
        f"Recurrent left-column gain from wave 1 to wave 4: col0={rec_col0_gain:.4f}, col1={rec_col1_gain:.4f}.",
        f"Ablated left-column gain from wave 1 to wave 4: col0={abl_col0_gain:.4f}, col1={abl_col1_gain:.4f}.",
        "Recurrent refinement appears active if recurrent gains exceed ablated gains on early columns.",
    ]

    if results["recurrent"]["delta_mean_by_wave"][1] is not None:
        findings.append(
            "Delta-based convergence remains questionable when recurrent delta stays high across late waves."
        )

    return findings


def expectation_summary(results):
    rec = results["recurrent"]
    abl = results["ablated"]

    rec_col0_gain = rec["accuracy_grid"][-1][0] - rec["accuracy_grid"][0][0]
    rec_col1_gain = rec["accuracy_grid"][-1][1] - rec["accuracy_grid"][0][1]
    abl_col0_gain = abl["accuracy_grid"][-1][0] - abl["accuracy_grid"][0][0]
    abl_col1_gain = abl["accuracy_grid"][-1][1] - abl["accuracy_grid"][0][1]

    recurrent_improves = rec_col0_gain > 0.10 and rec_col1_gain > 0.10
    ablated_flat = abs(abl_col0_gain) < 0.02 and abs(abl_col1_gain) < 0.02
    rightmost_strong = rec["accuracy_grid"][-1][-1] > 0.95
    convergence_supported = rec["delta_mean_by_wave"][3] < 0.5 * rec["delta_mean_by_wave"][1]

    return {
        "recurrent_improves_early_columns": {
            "expected": "large early-column gains with recurrence",
            "observed": f"col0 gain={rec_col0_gain:.4f}, col1 gain={rec_col1_gain:.4f}",
            "pass": recurrent_improves,
        },
        "ablated_stays_flat": {
            "expected": "little or no early-column gain without recurrence",
            "observed": f"col0 gain={abl_col0_gain:.4f}, col1 gain={abl_col1_gain:.4f}",
            "pass": ablated_flat,
        },
        "rightmost_final_accuracy": {
            "expected": "rightmost final accuracy above 0.95",
            "observed": f"{rec['accuracy_grid'][-1][-1]:.4f}",
            "pass": rightmost_strong,
        },
        "delta_convergence_signal": {
            "expected": "delta should shrink across later waves",
            "observed": f"wave2={rec['delta_mean_by_wave'][1]:.4f}, wave4={rec['delta_mean_by_wave'][3]:.4f}",
            "pass": convergence_supported,
        },
    }


def render_results_markdown(results, run_command):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config = results["config"]
    findings = evaluate_findings(results)
    expectations = expectation_summary(results)

    lines = [
        "# SRWN Experiment Results",
        "",
        f"Generated: {now}",
        "",
        "## What Was Run",
        "",
        "Command:",
        "",
        "```bash",
        run_command,
        "```",
        "",
        "Configuration:",
        "",
        "| Parameter | Value |",
        "|---|---|",
    ]

    for key in [
        "seed",
        "rows",
        "cols",
        "hidden_dim",
        "train_waves",
        "eval_waves",
        "learning_rate",
        "epochs",
        "batch_size",
        "train_size",
        "val_size",
        "test_size",
        "conf_threshold",
    ]:
        lines.append(f"| {key} | {config[key]} |")

    lines.extend([
        "",
        "## Recurrent Accuracy Grid",
        "",
        grid_to_markdown(results["recurrent"]["accuracy_grid"]),
        "",
        "## Ablated Accuracy Grid",
        "",
        grid_to_markdown(results["ablated"]["accuracy_grid"]),
        "",
        "## Key Metrics",
        "",
        "| Metric | Recurrent | Ablated |",
        "|---|---|---|",
        f"| Horizontal monotonicity | {results['recurrent']['monotonicity']['horizontal']:.4f} | {results['ablated']['monotonicity']['horizontal']:.4f} |",
        f"| Vertical monotonicity | {results['recurrent']['monotonicity']['vertical']:.4f} | {results['ablated']['monotonicity']['vertical']:.4f} |",
        f"| Halt mean wave | {results['recurrent']['halting']['mean_wave']:.4f} | {results['ablated']['halting']['mean_wave']:.4f} |",
        f"| Halted-policy accuracy | {results['recurrent']['halting']['halted_accuracy']:.4f} | {results['ablated']['halting']['halted_accuracy']:.4f} |",
        "",
        "## Expected vs Observed",
        "",
        "| Check | Expected | Observed | Status |",
        "|---|---|---|---|",
    ])

    for key, value in expectations.items():
        label = key.replace("_", " ")
        status = "PASS" if value["pass"] else "FAIL"
        lines.append(f"| {label} | {value['expected']} | {value['observed']} | {status} |")

    lines.extend([
        "",
        "## How To Interpret",
        "",
        "1. If recurrent early-column gains are high while ablated gains are near zero, the recurrent wave path is contributing meaningful refinement.",
        "2. If rightmost final accuracy is high but delta convergence fails, classification works but fixed-point convergence is not yet validated.",
        "3. Halted-policy accuracy close to rightmost final accuracy means confidence-threshold exit is viable on this task.",
        "4. For direct comparison against a conventional MLP baseline, read outputs/benchmark_results.md.",
        "",
        "## Evaluation",
        "",
    ])

    for idx, finding in enumerate(findings, start=1):
        lines.append(f"{idx}. {finding}")

    lines.append("")
    return "\n".join(lines)


def save_outputs(results, out_dir, run_command):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "results.json")
    md_path = os.path.join(out_dir, "results.md")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_results_markdown(results, run_command))

    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Run the SRWN parity ablation experiment.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Directory for saved results files.")
    parser.add_argument("--no-save", action="store_true", help="Do not write outputs/results.json and outputs/results.md.")
    args = parser.parse_args()

    results = run_experiment()
    run_command = "D:/apps/Python39/python.exe srwn_experiment.py --json"

    if not args.no_save:
        json_path, md_path = save_outputs(results, args.out_dir, run_command)
        results["saved"] = {
            "json": json_path,
            "markdown": md_path,
        }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()