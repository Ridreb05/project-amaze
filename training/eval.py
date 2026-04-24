"""
Evaluation Script
=================
Compares base (untrained) vs trained Qwen2.5-1.5B-Instruct on the
spacecraft rendezvous environment.

Generates:
  - eval_results.json  (quantitative comparison)
  - plots/eval_comparison.png  (bar chart for README)

Run after training:
    python training/eval.py --env-url https://your-space.hf.space \
                            --trained-model ./trained_model \
                            --n-episodes 20
"""

import os
import sys
import json
import argparse
import time
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import SpacecraftEnvClientSync
from models import RendezvousAction
from inference import build_prompt, parse_action, greedy_action, SYSTEM_PROMPT


def run_evaluation_episodes(
    env_url: str,
    n_episodes: int,
    agent_type: str,  # "greedy" | "llm_base" | "llm_trained"
    model=None,
    tokenizer=None,
    seeds: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Run N evaluation episodes and return per-episode results."""

    results = []
    difficulties = ["easy", "medium", "hard"]

    with SpacecraftEnvClientSync(base_url=env_url) as client:
        for i in range(n_episodes):
            seed = seeds[i] if seeds else (1000 + i)
            difficulty = difficulties[i % len(difficulties)]

            obs = client.reset(seed=seed, difficulty=difficulty)
            obs_dict = obs.model_dump()
            total_reward = 0.0
            steps = 0

            while not obs.done:
                steps += 1

                if agent_type == "greedy":
                    action_dict = greedy_action(obs_dict)

                elif agent_type in ("llm_base", "llm_trained") and model is not None:
                    prompt = build_prompt(obs_dict)
                    full_prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>\n"

                    inputs = tokenizer(
                        full_prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=1024,
                    ).to(model.device)

                    with __import__("torch").no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=200,
                            temperature=0.1,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id,
                        )

                    generated = tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                    action_dict = parse_action(generated) or greedy_action(obs_dict)
                else:
                    action_dict = greedy_action(obs_dict)

                action = RendezvousAction(**action_dict)
                resp = client.step(action)
                obs = resp.observation
                obs_dict = obs.model_dump()
                total_reward += resp.reward

            grade = client.grade()
            results.append({
                "episode": i,
                "seed": seed,
                "difficulty": difficulty,
                "agent_type": agent_type,
                "docked": grade.get("docked", False),
                "score": grade.get("score", 0.0),
                "steps": steps,
                "total_reward": round(total_reward, 4),
                "fuel_remaining_pct": grade.get("fuel_efficiency_pct", 0.0),
                "final_distance_m": grade.get("final_distance_m", 999.0),
            })

            print(
                f"  [{agent_type}] Episode {i+1}/{n_episodes} | "
                f"difficulty={difficulty} | docked={grade.get('docked')} | "
                f"reward={total_reward:.2f} | score={grade.get('score'):.3f}"
            )

    return results


def compute_metrics(results: List[Dict]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {}

    return {
        "n_episodes": n,
        "dock_rate": round(sum(1 for r in results if r["docked"]) / n, 3),
        "mean_reward": round(sum(r["total_reward"] for r in results) / n, 4),
        "mean_score": round(sum(r["score"] for r in results) / n, 4),
        "mean_steps": round(sum(r["steps"] for r in results) / n, 1),
        "mean_fuel_remaining_pct": round(
            sum(r["fuel_remaining_pct"] for r in results) / n, 1
        ),
        "mean_final_distance_m": round(
            sum(r["final_distance_m"] for r in results) / n, 2
        ),
        # Per-difficulty breakdown
        "by_difficulty": {
            diff: {
                "dock_rate": round(
                    sum(1 for r in results if r["difficulty"] == diff and r["docked"]) /
                    max(sum(1 for r in results if r["difficulty"] == diff), 1), 3
                ),
                "mean_reward": round(
                    sum(r["total_reward"] for r in results if r["difficulty"] == diff) /
                    max(sum(1 for r in results if r["difficulty"] == diff), 1), 4
                ),
            }
            for diff in ["easy", "medium", "hard"]
        }
    }


def plot_comparison(baseline_metrics: Dict, trained_metrics: Dict, output_path: str):
    """Generate comparison bar chart saved as PNG."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
        import numpy as np

        fig, axes = plt.subplots(1, 3, figsize=(12, 5))
        fig.suptitle(
            "Spacecraft Rendezvous: Baseline vs GRPO-Trained Agent",
            fontsize=14, fontweight="bold"
        )

        metrics = [
            ("Docking Success Rate", "dock_rate", "%", 100),
            ("Mean Episode Reward", "mean_reward", "", 1),
            ("Mean Final Distance (m)", "mean_final_distance_m", "m", 1),
        ]

        colors = ["#E74C3C", "#2ECC71"]  # red=baseline, green=trained

        for ax, (title, key, unit, scale) in zip(axes, metrics):
            base_val = baseline_metrics.get(key, 0) * scale
            train_val = trained_metrics.get(key, 0) * scale

            bars = ax.bar(
                ["Baseline\n(Untrained)", "GRPO Trained"],
                [base_val, train_val],
                color=colors,
                width=0.5,
                edgecolor="black",
                linewidth=0.8,
            )

            # Value labels on bars
            for bar, val in zip(bars, [base_val, train_val]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * max(base_val, train_val, 1),
                    f"{val:.1f}{unit}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold"
                )

            ax.set_title(title, fontsize=11)
            ax.set_ylabel(f"{unit}" if unit else "Value")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Improvement annotation
        if baseline_metrics.get("dock_rate", 0) < trained_metrics.get("dock_rate", 0):
            improvement = (
                (trained_metrics["dock_rate"] - baseline_metrics["dock_rate"]) * 100
            )
            fig.text(
                0.5, 0.02,
                f"GRPO training improved docking success by {improvement:.0f} percentage points",
                ha="center", fontsize=11, color="#2C3E50",
                style="italic"
            )

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved comparison plot: {output_path}")

    except ImportError:
        print("matplotlib not available — skipping plot generation")


def main():
    parser = argparse.ArgumentParser(description="Evaluate spacecraft rendezvous agents")
    parser.add_argument("--env-url", default="http://localhost:7860")
    parser.add_argument("--trained-model", default=None, help="Path to trained model dir")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--output", default="eval_results.json")
    parser.add_argument("--plot-output", default="assets/eval_comparison.png")
    args = parser.parse_args()

    seeds = list(range(2000, 2000 + args.n_episodes))

    print("=" * 60)
    print("SPACECRAFT RENDEZVOUS — AGENT EVALUATION")
    print("=" * 60)

    # ── Greedy baseline ────────────────────────────────────────────
    print(f"\n[1/2] Running greedy baseline ({args.n_episodes} episodes)...")
    baseline_results = run_evaluation_episodes(
        env_url=args.env_url,
        n_episodes=args.n_episodes,
        agent_type="greedy",
        seeds=seeds,
    )
    baseline_metrics = compute_metrics(baseline_results)

    print(f"\nBaseline metrics:")
    print(f"  Dock rate:    {baseline_metrics['dock_rate']*100:.0f}%")
    print(f"  Mean reward:  {baseline_metrics['mean_reward']:.3f}")
    print(f"  Mean score:   {baseline_metrics['mean_score']:.3f}")

    trained_results = []
    trained_metrics = {}

    # ── Trained model ──────────────────────────────────────────────
    if args.trained_model and os.path.exists(args.trained_model):
        print(f"\n[2/2] Running trained model from {args.trained_model} ({args.n_episodes} episodes)...")
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                args.trained_model,
                max_seq_length=1024,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)

            trained_results = run_evaluation_episodes(
                env_url=args.env_url,
                n_episodes=args.n_episodes,
                agent_type="llm_trained",
                model=model,
                tokenizer=tokenizer,
                seeds=seeds,
            )
            trained_metrics = compute_metrics(trained_results)

            print(f"\nTrained model metrics:")
            print(f"  Dock rate:    {trained_metrics['dock_rate']*100:.0f}%")
            print(f"  Mean reward:  {trained_metrics['mean_reward']:.3f}")
            print(f"  Mean score:   {trained_metrics['mean_score']:.3f}")

            improvement = (
                (trained_metrics["dock_rate"] - baseline_metrics["dock_rate"]) * 100
            )
            print(f"\n  Improvement: +{improvement:.0f} percentage points docking success")

        except ImportError as e:
            print(f"  Could not load trained model: {e}")
    else:
        print("\n[2/2] No trained model path provided — skipping trained evaluation")

    # ── Save results ───────────────────────────────────────────────
    output = {
        "baseline": {"metrics": baseline_metrics, "episodes": baseline_results},
        "trained": {"metrics": trained_metrics, "episodes": trained_results},
        "comparison": {
            "dock_rate_improvement": round(
                (trained_metrics.get("dock_rate", 0) - baseline_metrics.get("dock_rate", 0)) * 100, 1
            ) if trained_metrics else None,
            "reward_improvement": round(
                trained_metrics.get("mean_reward", 0) - baseline_metrics.get("mean_reward", 0), 4
            ) if trained_metrics else None,
        }
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved evaluation results: {args.output}")

    # ── Generate plot ──────────────────────────────────────────────
    if trained_metrics:
        plot_comparison(baseline_metrics, trained_metrics, args.plot_output)

    print("\nDone.")


if __name__ == "__main__":
    main()
