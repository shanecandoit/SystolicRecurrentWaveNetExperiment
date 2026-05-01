import argparse
import json
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from srwn_experiment import SRWN, build_parity_dataset, set_seed

TOTAL_BITS = 8
N_CLASSES = 2


class MLPBaseline(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(TOTAL_BITS, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, N_CLASSES)

    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        return self.fc3(h)


def accuracy(logits, labels):
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def count_params(model):
    return sum(param.numel() for param in model.parameters())


def flatten_column_input(column_input):
    return column_input.reshape(column_input.shape[0], -1)


def train_srwn(config, recurrent, train_x, train_y, val_x, val_y):
    model = SRWN(config["rows"], config["cols"], config["hidden_dim"], recurrent=recurrent)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    wave_weights = torch.linspace(0.5, 1.0, config["train_waves"])
    col_weights = torch.linspace(0.5, 1.0, config["cols"])
    total_weight = float((wave_weights[:, None] * col_weights[None, :]).sum().item())

    history = {"train_loss": [], "val_acc": []}
    best_val = -1.0
    best_state = None

    for _ in range(config["epochs"]):
        perm = torch.randperm(train_x.shape[0])
        model.train()
        total_loss = 0.0
        n_batches = 0

        for start in range(0, train_x.shape[0], config["batch_size"]):
            idx = perm[start:start + config["batch_size"]]
            batch_x = train_x[idx]
            batch_y = train_y[idx]

            optimizer.zero_grad()
            logits, _confs, _deltas = model(batch_x, config["train_waves"])
            loss = 0.0
            for wave in range(config["train_waves"]):
                for col in range(config["cols"]):
                    weight = wave_weights[wave] * col_weights[col]
                    loss = loss + weight * F.cross_entropy(logits[:, wave, col, :], batch_y)
            norm_loss = loss / total_weight
            norm_loss.backward()
            optimizer.step()

            total_loss += norm_loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_logits, _confs, _deltas = model(val_x, config["eval_waves"])
            val_acc = accuracy(val_logits[:, -1, -1, :], val_y)

        history["train_loss"].append(total_loss / max(1, n_batches))
        history["val_acc"].append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, history


def train_mlp(config, train_x_flat, train_y, val_x_flat, val_y):
    model = MLPBaseline(config["mlp_hidden_dim"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    history = {"train_loss": [], "val_acc": []}
    best_val = -1.0
    best_state = None

    for _ in range(config["epochs"]):
        perm = torch.randperm(train_x_flat.shape[0])
        model.train()
        total_loss = 0.0
        n_batches = 0

        for start in range(0, train_x_flat.shape[0], config["batch_size"]):
            idx = perm[start:start + config["batch_size"]]
            batch_x = train_x_flat[idx]
            batch_y = train_y[idx]

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x_flat)
            val_acc = accuracy(val_logits, val_y)

        history["train_loss"].append(total_loss / max(1, n_batches))
        history["val_acc"].append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, history


def eval_srwn(model, x, y, conf_threshold, eval_waves):
    model.eval()
    with torch.no_grad():
        logits, confs, _deltas = model(x, eval_waves)
        rightmost_acc = accuracy(logits[:, -1, -1, :], y)

        best_conf = confs.max(dim=2).values
        best_col = confs.argmax(dim=2)
        reached = best_conf >= conf_threshold
        first_wave = reached.float().argmax(dim=1) + 1
        first_wave = torch.where(reached.any(dim=1), first_wave, torch.full_like(first_wave, eval_waves))
        wave_idx = first_wave - 1
        sample_idx = torch.arange(y.shape[0])
        cols = best_col[sample_idx, wave_idx]
        halted_logits = logits[sample_idx, wave_idx, cols, :]
        halted_acc = accuracy(halted_logits, y)
        mean_halt = first_wave.float().mean().item()

    return {
        "rightmost_test_accuracy": round(rightmost_acc, 4),
        "halted_test_accuracy": round(halted_acc, 4),
        "mean_halt_wave": round(mean_halt, 4),
    }


def eval_mlp(model, x_flat, y):
    model.eval()
    with torch.no_grad():
        logits = model(x_flat)
    return {"test_accuracy": round(accuracy(logits, y), 4)}


def srwn_macs(config, mean_halt_wave):
    bits_per_col = TOTAL_BITS // config["cols"]
    input_projection = config["cols"] * bits_per_col * config["hidden_dim"]
    per_node = 3 * config["hidden_dim"] * config["hidden_dim"]
    node_cost = config["rows"] * config["cols"] * per_node
    head_cost = config["cols"] * config["hidden_dim"] * N_CLASSES
    per_wave = node_cost + head_cost
    fixed = input_projection + config["eval_waves"] * per_wave
    adaptive = input_projection + mean_halt_wave * per_wave
    return {
        "fixed_eval_macs": round(float(fixed), 2),
        "adaptive_estimated_macs": round(float(adaptive), 2),
        "per_wave_macs": round(float(per_wave), 2),
    }


def mlp_macs(config):
    h = config["mlp_hidden_dim"]
    macs = TOTAL_BITS * h + h * h + h * N_CLASSES
    return {"fixed_eval_macs": round(float(macs), 2)}


def plot_history(results, out_dir):
    epochs = list(range(1, len(results["srwn_recurrent"]["history"]["train_loss"]) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, results["srwn_recurrent"]["history"]["train_loss"], label="SRWN recurrent")
    plt.plot(epochs, results["srwn_ablated"]["history"]["train_loss"], label="SRWN ablated")
    plt.plot(epochs, results["mlp_baseline"]["history"]["train_loss"], label="MLP baseline")
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Train loss by model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_loss.png"), dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, results["srwn_recurrent"]["history"]["val_acc"], label="SRWN recurrent")
    plt.plot(epochs, results["srwn_ablated"]["history"]["val_acc"], label="SRWN ablated")
    plt.plot(epochs, results["mlp_baseline"]["history"]["val_acc"], label="MLP baseline")
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title("Validation accuracy by model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "val_accuracy.png"), dpi=140)
    plt.close()


def plot_compute_tradeoff(results, out_dir):
    labels = [
        "SRWN recurrent fixed",
        "SRWN recurrent adaptive",
        "SRWN ablated fixed",
        "MLP baseline",
    ]
    x = [
        results["srwn_recurrent"]["compute"]["fixed_eval_macs"],
        results["srwn_recurrent"]["compute"]["adaptive_estimated_macs"],
        results["srwn_ablated"]["compute"]["fixed_eval_macs"],
        results["mlp_baseline"]["compute"]["fixed_eval_macs"],
    ]
    y = [
        results["srwn_recurrent"]["test"]["rightmost_test_accuracy"],
        results["srwn_recurrent"]["test"]["halted_test_accuracy"],
        results["srwn_ablated"]["test"]["rightmost_test_accuracy"],
        results["mlp_baseline"]["test"]["test_accuracy"],
    ]

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=90)
    for i, label in enumerate(labels):
        plt.annotate(label, (x[i], y[i]), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Estimated MACs per sample")
    plt.ylabel("Test accuracy")
    plt.title("Accuracy vs compute")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_vs_compute.png"), dpi=140)
    plt.close()


def render_markdown(results):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rec = results["srwn_recurrent"]
    abl = results["srwn_ablated"]
    mlp = results["mlp_baseline"]

    lines = [
        "# SRWN Baseline Comparison",
        "",
        f"Generated: {now}",
        "",
        "## What was run",
        "",
        "- Trained SRWN recurrent, SRWN ablated, and MLP baseline on parity data.",
        "- Logged train loss and validation accuracy by epoch.",
        "- Evaluated test accuracy and estimated MACs.",
        "",
        "## Final comparison",
        "",
        "| Model | Test metric | Params | Estimated MACs/sample |",
        "|---|---|---|---|",
        f"| SRWN recurrent fixed | acc={rec['test']['rightmost_test_accuracy']:.4f} | {rec['params']} | {rec['compute']['fixed_eval_macs']:.2f} |",
        f"| SRWN recurrent adaptive | acc={rec['test']['halted_test_accuracy']:.4f} | {rec['params']} | {rec['compute']['adaptive_estimated_macs']:.2f} |",
        f"| SRWN ablated fixed | acc={abl['test']['rightmost_test_accuracy']:.4f} | {abl['params']} | {abl['compute']['fixed_eval_macs']:.2f} |",
        f"| MLP baseline | acc={mlp['test']['test_accuracy']:.4f} | {mlp['params']} | {mlp['compute']['fixed_eval_macs']:.2f} |",
        "",
        "## How to interpret",
        "",
    ]

    if rec["test"]["halted_test_accuracy"] >= mlp["test"]["test_accuracy"]:
        lines.append("1. SRWN adaptive mode matches or beats MLP accuracy on this task.")
    else:
        lines.append("1. MLP still beats SRWN adaptive mode on this task.")

    if rec["compute"]["adaptive_estimated_macs"] < mlp["compute"]["fixed_eval_macs"]:
        lines.append("2. SRWN adaptive mode is compute-cheaper than MLP at its measured halt behavior.")
    else:
        lines.append("2. SRWN adaptive mode is still compute-heavier than MLP with this configuration.")

    if rec["test"]["rightmost_test_accuracy"] > abl["test"]["rightmost_test_accuracy"]:
        lines.append("3. Recurrent SRWN beats ablated SRWN, so recurrence contributes to final quality.")
    else:
        lines.append("3. Recurrent SRWN does not beat ablated SRWN in final accuracy under this setup.")

    lines.extend([
        "",
        "## Plot files",
        "",
        "- outputs/train_loss.png",
        "- outputs/val_accuracy.png",
        "- outputs/accuracy_vs_compute.png",
        "",
    ])

    return "\n".join(lines)


def run(config, out_dir):
    set_seed(config["seed"])

    train_col, train_y = build_parity_dataset(config["train_size"], config["cols"], config["seed"])
    val_col, val_y = build_parity_dataset(config["val_size"], config["cols"], config["seed"] + 1)
    test_col, test_y = build_parity_dataset(config["test_size"], config["cols"], config["seed"] + 2)

    train_flat = flatten_column_input(train_col)
    val_flat = flatten_column_input(val_col)
    test_flat = flatten_column_input(test_col)

    srwn_rec, hist_rec = train_srwn(config, True, train_col, train_y, val_col, val_y)
    srwn_abl, hist_abl = train_srwn(config, False, train_col, train_y, val_col, val_y)
    mlp_model, hist_mlp = train_mlp(config, train_flat, train_y, val_flat, val_y)

    rec_test = eval_srwn(srwn_rec, test_col, test_y, config["conf_threshold"], config["eval_waves"])
    abl_test = eval_srwn(srwn_abl, test_col, test_y, config["conf_threshold"], config["eval_waves"])
    mlp_test = eval_mlp(mlp_model, test_flat, test_y)

    rec_compute = srwn_macs(config, rec_test["mean_halt_wave"])
    abl_compute = srwn_macs(config, abl_test["mean_halt_wave"])

    results = {
        "config": config,
        "srwn_recurrent": {
            "history": hist_rec,
            "test": rec_test,
            "params": count_params(srwn_rec),
            "compute": rec_compute,
        },
        "srwn_ablated": {
            "history": hist_abl,
            "test": abl_test,
            "params": count_params(srwn_abl),
            "compute": abl_compute,
        },
        "mlp_baseline": {
            "history": hist_mlp,
            "test": mlp_test,
            "params": count_params(mlp_model),
            "compute": mlp_macs(config),
        },
    }

    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "benchmark_metrics.json")
    md_path = os.path.join(out_dir, "benchmark_results.md")

    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(results))

    plot_history(results, out_dir)
    plot_compute_tradeoff(results, out_dir)

    return {
        "metrics": metrics_path,
        "markdown": md_path,
        "train_loss_plot": os.path.join(out_dir, "train_loss.png"),
        "val_accuracy_plot": os.path.join(out_dir, "val_accuracy.png"),
        "accuracy_vs_compute_plot": os.path.join(out_dir, "accuracy_vs_compute.png"),
        "summary": {
            "srwn_recurrent": results["srwn_recurrent"]["test"],
            "srwn_ablated": results["srwn_ablated"]["test"],
            "mlp_baseline": results["mlp_baseline"]["test"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="SRWN benchmark with train/test plots and MLP comparison")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=12)
    parser.add_argument("--mlp-hidden-dim", type=int, default=24)
    parser.add_argument("--train-waves", type=int, default=4)
    parser.add_argument("--eval-waves", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=1024)
    parser.add_argument("--test-size", type=int, default=2048)
    parser.add_argument("--conf-threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quick", action="store_true", help="Fast smoke run with fewer epochs.")
    args = parser.parse_args()

    config = {
        "seed": args.seed,
        "rows": args.rows,
        "cols": args.cols,
        "hidden_dim": args.hidden_dim,
        "mlp_hidden_dim": args.mlp_hidden_dim,
        "train_waves": args.train_waves,
        "eval_waves": args.eval_waves,
        "learning_rate": args.learning_rate,
        "epochs": 30 if args.quick else args.epochs,
        "batch_size": args.batch_size,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "conf_threshold": args.conf_threshold,
    }

    outputs = run(config, args.out_dir)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
