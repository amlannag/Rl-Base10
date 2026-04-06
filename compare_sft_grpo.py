"""
Compare SFT runs at different dataset sizes.
Plots SFT accuracy vs number of training examples.

Usage:
    uv run python compare_sft_grpo.py \
        --sft_dirs output/sft-100 output/sft-250 output/sft-500 output/sft-750 \
        --output_dir output/comparison
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def load_sft_results(sft_dirs: list) -> list:
    results = []
    for d in sft_dirs:
        path = os.path.join(d, "results.json")
        if os.path.exists(path):
            with open(path) as f:
                results.append(json.load(f))
        else:
            print(f"Warning: No results.json found in {d}")
    return sorted(results, key=lambda x: x["num_examples"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_dirs", nargs="+", default=["output/sft-100", "output/sft-250", "output/sft-500", "output/sft-750"])
    parser.add_argument("--output_dir", type=str, default="output/comparison")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sft_results = load_sft_results(args.sft_dirs)
    if not sft_results:
        print("No SFT results found. Run sft_train.py first.")
        return

    examples = [r["num_examples"] for r in sft_results]
    pre_acc = [r["pre_accuracy"] for r in sft_results]
    post_acc = [r["post_accuracy"] for r in sft_results]

    plt.style.use("bmh")
    pdf_path = os.path.join(args.output_dir, "sft_comparison.pdf")

    with PdfPages(pdf_path) as pdf:

        # ── Plot 1: Post-SFT accuracy vs number of examples ─────────────────
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(examples, post_acc, "o-", color="#2ecc71", linewidth=2.5, markersize=8, label="SFT (post-training)")
        ax.axhline(y=pre_acc[0], color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Base model ({pre_acc[0]:.1f}%)")

        for x, y in zip(examples, post_acc):
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center")

        ax.set_xlabel("Number of SFT Training Examples", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("SFT Data Efficiency: Accuracy vs Training Examples", fontsize=14, pad=15)
        ax.legend()
        ax.grid(True, alpha=0.3)
        pdf.savefig(bbox_inches="tight")
        plt.close()

        # ── Plot 2: Pre vs Post accuracy per run ─────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.35
        x = range(len(examples))

        ax.bar([i - width/2 for i in x], pre_acc, width, label="Before SFT", color="#e74c3c", alpha=0.8)
        ax.bar([i + width/2 for i in x], post_acc, width, label="After SFT", color="#2ecc71", alpha=0.8)

        ax.set_xticks(list(x))
        ax.set_xticklabels([str(e) + " examples" for e in examples])
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("Before vs After SFT by Dataset Size", fontsize=14, pad=15)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        pdf.savefig(bbox_inches="tight")
        plt.close()

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("SFT RESULTS SUMMARY")
    print("="*50)
    print(f"{'Examples':>10} | {'Pre-SFT':>10} | {'Post-SFT':>10} | {'Gain':>8}")
    print("-"*50)
    for r in sft_results:
        gain = r["post_accuracy"] - r["pre_accuracy"]
        print(f"{r['num_examples']:>10} | {r['pre_accuracy']:>9.2f}% | {r['post_accuracy']:>9.2f}% | {gain:>+7.2f}%")

    print(f"\nPlots saved to {pdf_path}")


if __name__ == "__main__":
    main()
