import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt

from fashion_mnist_benchmark import (
    cnn_macs,
    count_params,
    eval_cnn,
    eval_srwn,
    make_loaders,
    set_seed,
    srwn_macs,
    train_cnn,
    train_srwn,
)


def build_srwn_configs():
    """Wide SRWN variants to find accuracy ceiling. All use rows=3, cols=4, waves=3."""
    return [
        {
            "name": "wide_r3_h64_w3",
            "family": "wide",
            "rows": 3,
            "cols": 4,
            "hidden_dim": 64,
            "train_waves": 3,
            "eval_waves": 3,
        },
        {
            "name": "wide_r3_h96_w3",
            "family": "wide",
            "rows": 3,
            "cols": 4,
            "hidden_dim": 96,
            "train_waves": 3,
            "eval_waves": 3,
        },
        {
            "name": "wide_r3_h128_w3",
            "family": "wide",
            "rows": 3,
            "cols": 4,
            "hidden_dim": 128,
            "train_waves": 3,
            "eval_waves": 3,
        },
        {
            "name": "wide_r3_h160_w3",
            "family": "wide",
            "rows": 3,
            "cols": 4,
            "hidden_dim": 160,
            "train_waves": 3,
            "eval_waves": 3,
        },
        {
            "name": "wide_r3_h192_w3",
            "family": "wide",
            "rows": 3,
            "cols": 4,
            "hidden_dim": 192,
            "train_waves": 3,
            "eval_waves": 3,
        },
    ]


def plot_accuracy_bar(metrics, out_dir):
    names = [entry["name"] for entry in metrics["srwn"]] + ["cnn_baseline"]
    values = [entry["test"]["rightmost_test_accuracy"] for entry in metrics["srwn"]] + [metrics["cnn"]["test"]["test_accuracy"]]
    colors = ["#1f77b4"] * len(metrics["srwn"]) + ["#d62728"]

    plt.figure(figsize=(11, 5))
    plt.bar(names, values, color=colors)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Test accuracy")
    plt.title("Fashion-MNIST v3: Wide SRWN variants vs CNN (accuracy)")
    plt.axhline(y=metrics["cnn"]["test"]["test_accuracy"], color="red", linestyle="--", alpha=0.7, label=f"CNN baseline: {metrics['cnn']['test']['test_accuracy']:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fashion_v3_accuracy_bar.png"), dpi=140)
    plt.close()


def plot_compute_bar(metrics, out_dir):
    names = [entry["name"] for entry in metrics["srwn"]] + ["cnn_baseline"]
    values = [entry["compute"]["adaptive_estimated_macs"] for entry in metrics["srwn"]] + [metrics["cnn"]["compute"]["fixed_eval_macs"]]
    colors = ["#1f77b4"] * len(metrics["srwn"]) + ["#d62728"]

    plt.figure(figsize=(11, 5))
    plt.bar(names, values, color=colors)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Estimated MACs per sample")
    plt.title("Fashion-MNIST v3: Wide SRWN variants vs CNN (compute)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fashion_v3_compute_bar.png"), dpi=140)
    plt.close()


def plot_acc_vs_macs(metrics, out_dir):
    plt.figure(figsize=(8, 5))

    for entry in metrics["srwn"]:
        x = entry["compute"]["adaptive_estimated_macs"]
        y = entry["test"]["rightmost_test_accuracy"]
        plt.scatter([x], [y], s=100, marker="o", color="#1f77b4", alpha=0.7)
        plt.annotate(entry["name"], (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)

    cnn_x = metrics["cnn"]["compute"]["fixed_eval_macs"]
    cnn_y = metrics["cnn"]["test"]["test_accuracy"]
    plt.scatter([cnn_x], [cnn_y], s=150, marker="*", color="#d62728", zorder=5)
    plt.annotate("cnn_baseline", (cnn_x, cnn_y), textcoords="offset points", xytext=(5, 5), fontsize=10, fontweight="bold")

    plt.xlabel("Estimated MACs per sample")
    plt.ylabel("Test accuracy")
    plt.title("Fashion-MNIST v3: Accuracy vs compute (wide SRWN push)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fashion_v3_acc_vs_macs.png"), dpi=140)
    plt.close()


def make_verdict(metrics):
    cnn_acc = metrics["cnn"]["test"]["test_accuracy"]
    cnn_macs = metrics["cnn"]["compute"]["fixed_eval_macs"]

    # For v3, focus on accuracy beating, not MACs
    beats_cnn_accuracy = []
    for entry in metrics["srwn"]:
        acc = entry["test"]["rightmost_test_accuracy"]
        if acc >= cnn_acc:
            beats_cnn_accuracy.append(entry)

    # Still track both-metric winners for completeness
    beats_both = []
    for entry in metrics["srwn"]:
        acc = entry["test"]["rightmost_test_accuracy"]
        macs = entry["compute"]["adaptive_estimated_macs"]
        if acc >= cnn_acc and macs < cnn_macs:
            beats_both.append(entry)

    best_acc = max(metrics["srwn"], key=lambda e: e["test"]["rightmost_test_accuracy"])
    best_eff = max(metrics["srwn"], key=lambda e: e["test"]["rightmost_test_accuracy"] / e["compute"]["adaptive_estimated_macs"])

    return {
        "cnn": {
            "test_accuracy": cnn_acc,
            "macs": cnn_macs,
        },
        "beats_cnn_on_accuracy": beats_cnn_accuracy,
        "beats_cnn_on_accuracy_and_macs": beats_both,
        "best_srwn_accuracy": best_acc,
        "best_srwn_efficiency": best_eff,
    }


def render_markdown(metrics, verdict):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Fashion-MNIST v3: Wide SRWN Push to Match CNN Accuracy",
        "",
        f"Generated: {now}",
        "",
        "## Goal",
        "",
        "Push SRWN to maximum width (hidden_dim) to find accuracy ceiling and determine if pure accuracy can match CNN.",
        "",
        "## CNN baseline",
        "",
        f"- Test accuracy: {verdict['cnn']['test_accuracy']:.4f}",
        f"- Estimated MACs/sample: {verdict['cnn']['macs']:.2f}",
        "",
        "## SRWN variant table (all h64→h192, rows=3, cols=4, waves=3)",
        "",
        "| Name | Hidden | Test acc | Halted acc | Mean halt wave | Adaptive MACs | Params | Beats CNN acc? |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    cnn_acc = verdict["cnn"]["test_accuracy"]
    for entry in metrics["srwn"]:
        cfg = entry["config"]
        test = entry["test"]
        comp = entry["compute"]
        beats_cnn = "✓ YES" if test["rightmost_test_accuracy"] >= cnn_acc else "✗ no"
        lines.append(
            f"| {entry['name']} | {cfg['hidden_dim']} | "
            f"{test['rightmost_test_accuracy']:.4f} | {test['halted_test_accuracy']:.4f} | {test['mean_halt_wave']:.4f} | "
            f"{comp['adaptive_estimated_macs']:.2f} | {entry['params']} | {beats_cnn} |"
        )

    lines.extend([
        "",
        "## Key findings",
        "",
    ])

    if verdict["beats_cnn_on_accuracy"]:
        lines.extend([
            f"✓ **Accuracy milestone achieved**: {len(verdict['beats_cnn_on_accuracy'])} SRWN variant(s) matched or exceeded CNN accuracy ({cnn_acc:.4f}):",
            "",
        ])
        for entry in verdict["beats_cnn_on_accuracy"]:
            lines.append(f"  - {entry['name']}: {entry['test']['rightmost_test_accuracy']:.4f}")
    else:
        best = verdict["best_srwn_accuracy"]
        gap = cnn_acc - best["test"]["rightmost_test_accuracy"]
        lines.append(f"✗ No SRWN variant reached CNN accuracy ({cnn_acc:.4f}).")
        lines.append(f"   Best achieved: {best['name']} at {best['test']['rightmost_test_accuracy']:.4f} (gap: {gap:.4f})")

    lines.extend([
        "",
        f"Best SRWN by accuracy: {verdict['best_srwn_accuracy']['name']} ({verdict['best_srwn_accuracy']['test']['rightmost_test_accuracy']:.4f})",
        f"Best SRWN by accuracy-per-MAC: {verdict['best_srwn_efficiency']['name']}",
        "",
        "## Plot files",
        "",
        "- outputs/fashion_v3_accuracy_bar.png",
        "- outputs/fashion_v3_compute_bar.png",
        "- outputs/fashion_v3_acc_vs_macs.png",
        "",
    ])

    if verdict["beats_cnn_on_accuracy_and_macs"]:
        lines.extend([
            "## Bonus: variants that beat CNN on both accuracy AND MACs",
            "",
        ])
        for entry in verdict["beats_cnn_on_accuracy_and_macs"]:
            lines.append(f"- {entry['name']}")
        lines.append("")

    return "\n".join(lines)


def run(config):
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader = make_loaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        train_size=config["train_size"],
        val_size=config["val_size"],
        test_size=config["test_size"],
        seed=config["seed"],
    )

    cnn_cfg = {
        "learning_rate": config["learning_rate"],
        "epochs": config["epochs"],
    }
    cnn_model, cnn_history = train_cnn(cnn_cfg, train_loader, val_loader, device)
    cnn_test = eval_cnn(cnn_model, test_loader, device)

    srwn_metrics = []
    for variant in build_srwn_configs():
        print(f"Training {variant['name']}...")
        srwn_cfg = {
            "rows": variant["rows"],
            "cols": variant["cols"],
            "hidden_dim": variant["hidden_dim"],
            "train_waves": variant["train_waves"],
            "eval_waves": variant["eval_waves"],
            "learning_rate": config["learning_rate"],
            "epochs": config["epochs"],
        }
        model, history = train_srwn(srwn_cfg, train_loader, val_loader, device)
        test = eval_srwn(model, test_loader, srwn_cfg["eval_waves"], config["conf_threshold"], device)
        srwn_metrics.append(
            {
                "name": variant["name"],
                "family": variant["family"],
                "config": srwn_cfg,
                "history": history,
                "test": test,
                "params": count_params(model),
                "compute": srwn_macs(srwn_cfg, test["mean_halt_wave"]),
            }
        )

    metrics = {
        "config": {**config, "device": device},
        "cnn": {
            "history": cnn_history,
            "test": cnn_test,
            "params": count_params(cnn_model),
            "compute": cnn_macs(),
        },
        "srwn": srwn_metrics,
    }

    verdict = make_verdict(metrics)

    os.makedirs(config["out_dir"], exist_ok=True)
    metrics_path = os.path.join(config["out_dir"], "fashion_v3_metrics.json")
    verdict_path = os.path.join(config["out_dir"], "fashion_v3_verdict.json")
    md_path = os.path.join(config["out_dir"], "fashion_v3_results.md")

    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with open(verdict_path, "w", encoding="utf-8") as handle:
        json.dump(verdict, handle, indent=2)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(metrics, verdict))

    plot_accuracy_bar(metrics, config["out_dir"])
    plot_compute_bar(metrics, config["out_dir"])
    plot_acc_vs_macs(metrics, config["out_dir"])

    return {
        "metrics": metrics_path,
        "verdict": verdict_path,
        "markdown": md_path,
        "plots": {
            "accuracy_bar": os.path.join(config["out_dir"], "fashion_v3_accuracy_bar.png"),
            "compute_bar": os.path.join(config["out_dir"], "fashion_v3_compute_bar.png"),
            "acc_vs_macs": os.path.join(config["out_dir"], "fashion_v3_acc_vs_macs.png"),
        },
        "headline": {
            "cnn": verdict["cnn"],
            "best_srwn_accuracy": {
                "name": verdict["best_srwn_accuracy"]["name"],
                "test_accuracy": verdict["best_srwn_accuracy"]["test"]["rightmost_test_accuracy"],
            },
            "beats_cnn_on_accuracy_count": len(verdict["beats_cnn_on_accuracy"]),
            "beats_cnn_on_both_count": len(verdict["beats_cnn_on_accuracy_and_macs"]),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Fashion-MNIST v3: wide SRWN push to match CNN accuracy")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--data-dir", type=str, default="outputs/fashion_data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=12000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--conf-threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    set_seed(args.seed)
    config = {
        "out_dir": args.out_dir,
        "data_dir": args.data_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "learning_rate": args.learning_rate,
        "conf_threshold": args.conf_threshold,
        "seed": args.seed,
    }

    outputs = run(config)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
