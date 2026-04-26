"""
inference.py — OpenEnv Submission Script
==========================================
Required by OpenEnv spec. Runs the agent against the environment
and emits structured stdout output consumed by the validator.

Output format (flushed immediately):
    [START] task=<task_name>
    [STEP] step=<n> reward=<r>
    [END] task=<task_name> score=<s> steps=<n>

Environment variables:
    HF_TOKEN        HuggingFace API key
    API_BASE_URL    LLM endpoint (default: HF router)
    MODEL_NAME      Model identifier
    ENV_BASE_URL    OpenEnv server URL (default: http://localhost:7860)
"""

import os
import sys
import json
import time
import re
from typing import Optional

# ---------------------------------------------------------------------------
# Stdout logging helpers (validator reads ONLY stdout)
# ---------------------------------------------------------------------------

def emit(line: str) -> None:
    print(line, flush=True)

def emit_start(task: str) -> None:
    emit(f"[START] task={task}")

def emit_step(step: int, reward: float) -> None:
    emit(f"[STEP] step={step} reward={reward:.4f}")

def emit_end(task: str, score: float, steps: int) -> None:
    emit(f"[END] task={task} score={score:.4f} steps={steps}")


# ---------------------------------------------------------------------------
# LLM action parser
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an autonomous spacecraft flight controller.
Your mission: guide a chaser spacecraft to safely rendezvous and dock with a target.

PHYSICS CONSTRAINTS:
- Fuel is limited. Every thruster firing costs propellant. Plan ahead.
- Approach from within ±45° of the docking axis (along-track direction).
- During communication blackouts, sensors go offline. Use last known state.
- Terminal docking requires: distance < 0.5m AND speed < 0.05 m/s.

OUTPUT FORMAT (JSON only, no markdown):
{
  "fx": <float between -2.0 and 2.0>,
  "fy": <float between -2.0 and 2.0>,
  "reasoning": "<your flight controller reasoning>"
}

STRATEGY:
- Far range (>50m): apply moderate burns in the direction of the target
- Mid range (10-50m): reduce speed, align with docking axis
- Close range (<10m): minimal thrust, bleed off velocity, approach slowly
- Blackout: reduce thrust, hold current trajectory, prepare for sensor return"""

def build_prompt(obs_dict: dict) -> str:
    """Format observation as natural language prompt for the LLM."""
    msg = obs_dict.get("message", "")
    comms = obs_dict.get("comms_active", True)

    if not comms:
        return (
            f"FLIGHT CONTROLLER STATUS — COMMUNICATIONS BLACKOUT\n\n"
            f"{msg}\n\n"
            f"Fuel remaining: {obs_dict.get('fuel_pct', 0):.0f}%\n"
            f"Step {obs_dict.get('step', 0)}/{obs_dict.get('max_steps', 100)}\n\n"
            f"Issue your thruster command. During blackout, minimize thrust."
        )

    los_warn = " ⚠ WARNING: LOS CONE VIOLATION — adjust approach angle!" if obs_dict.get("los_violation") else ""
    fuel_warn = " ⚠ LOW FUEL — conserve!" if obs_dict.get("fuel_pct", 100) < 25 else ""

    return (
        f"FLIGHT CONTROLLER STATUS\n\n"
        f"{msg}{los_warn}{fuel_warn}\n\n"
        f"SENSOR READINGS:\n"
        f"  Position: x={obs_dict.get('x_m', 0):.2f}m (radial), "
        f"y={obs_dict.get('y_m', 0):.2f}m (along-track)\n"
        f"  Velocity: vx={obs_dict.get('vx_ms', 0):.4f} m/s, "
        f"vy={obs_dict.get('vy_ms', 0):.4f} m/s\n"
        f"  Distance to target: {obs_dict.get('estimated_distance_m', 0):.2f}m\n"
        f"  Closing speed: {obs_dict.get('estimated_speed_ms', 0):.4f} m/s\n"
        f"  Approach angle: {obs_dict.get('los_angle_deg', 0):.1f}°\n\n"
        f"RESOURCES:\n"
        f"  Fuel: {obs_dict.get('fuel_kg', 0):.3f}kg ({obs_dict.get('fuel_pct', 0):.0f}%)\n"
        f"  Steps remaining: {obs_dict.get('steps_remaining', 0)}\n\n"
        f"Issue your next thruster command as JSON."
    )


def parse_action(text: str) -> Optional[dict]:
    """Parse LLM output as JSON action. Returns None on failure."""
    # Try direct JSON parse
    try:
        text = text.strip()
        # Strip markdown code fences
        text = re.sub(r"```(?:json)?\n?", "", text).strip()
        data = json.loads(text)
        return {
            "fx": float(data.get("fx", 0.0)),
            "fy": float(data.get("fy", 0.0)),
            "reasoning": str(data.get("reasoning", ""))[:500],
        }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass

    # Fallback: regex extraction
    try:
        fx_match = re.search(r'"fx"\s*:\s*(-?[\d.]+)', text)
        fy_match = re.search(r'"fy"\s*:\s*(-?[\d.]+)', text)
        if fx_match and fy_match:
            return {
                "fx": float(fx_match.group(1)),
                "fy": float(fy_match.group(1)),
                "reasoning": "Parsed via regex fallback.",
            }
    except (AttributeError, ValueError):
        pass

    return None


def greedy_action(obs: dict) -> dict:
    """
    Deterministic greedy fallback: thrust toward target.
    Used when LLM output is unparseable.
    """
    import math

    if not obs.get("comms_active", True):
        return {"fx": 0.0, "fy": 0.0, "reasoning": "Blackout: holding trajectory."}

    x = obs.get("x_m", 0.0)
    y = obs.get("y_m", 0.0)
    dist = obs.get("estimated_distance_m", 1.0)

    if dist < 1e-3:
        return {"fx": 0.0, "fy": 0.0, "reasoning": "At target — zero thrust."}

    # Scale thrust: strong far, gentle close
    scale = min(1.5, max(0.1, dist / 30.0))

    fx = (-x / dist) * scale
    fy = (-y / dist) * scale

    return {
        "fx": round(max(-2.0, min(2.0, fx)), 3),
        "fy": round(max(-2.0, min(2.0, fy)), 3),
        "reasoning": f"Greedy: thrusting toward target at {dist:.1f}m.",
    }


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_episode(
    env_url: str,
    task_name: str,
    seed: int,
    difficulty: str,
    llm_client=None,
    model_name: str = "",
) -> float:
    """Run one episode and return the grade score."""
    import httpx

    http = httpx.Client(base_url=env_url, timeout=60.0)

    # Reset
    reset_resp = http.post(
        "/reset",
        json={"seed": seed, "difficulty": difficulty},
    )
    reset_resp.raise_for_status()
    obs = reset_resp.json()

    emit_start(task_name)

    step_num = 0
    cumulative_reward = 0.0

    while not obs.get("done", False):
        step_num += 1

        # Build action
        if llm_client is not None:
            try:
                prompt = build_prompt(obs)
                response = llm_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=256,
                    temperature=0.1,
                )
                action_text = response.choices[0].message.content
                action = parse_action(action_text)
            except Exception as e:
                action = None
        else:
            action = None

        # Fallback to greedy if LLM failed
        if action is None:
            action = greedy_action(obs)

        # Step
        step_resp = http.post("/step", json={"action": action})
        if step_resp.status_code != 200:
            break

        step_data = step_resp.json()
        reward = step_data.get("reward", 0.0)
        cumulative_reward += reward
        obs = step_data.get("observation", obs)

        emit_step(step_num, reward)

    # Grade
    grade_resp = http.post("/grade")
    grade_resp.raise_for_status()
    grade = grade_resp.json()
    score = grade.get("score", 0.1)

    emit_end(task_name, score, step_num)
    http.close()
    return score


def main():
    """Entry point — runs inference against all difficulty levels."""
    env_url = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
    hf_token = os.environ.get("HF_TOKEN", os.environ.get("API_KEY", ""))
    api_base = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

    # Set up LLM client if token available
    llm_client = None
    if hf_token:
        try:
            from openai import OpenAI
            llm_client = OpenAI(api_key=hf_token, base_url=api_base)
        except ImportError:
            emit("[WARN] openai package not installed — using greedy fallback")
    else:
        emit("[WARN] No HF_TOKEN — using greedy baseline agent")

    # Run episodes
    scenarios = [
        ("rendezvous_easy", 42, "easy"),
        ("rendezvous_medium", 123, "medium"),
        ("rendezvous_hard", 777, "hard"),
    ]

    scores = []
    for task_name, seed, difficulty in scenarios:
        try:
            score = run_episode(
                env_url=env_url,
                task_name=task_name,
                seed=seed,
                difficulty=difficulty,
                llm_client=llm_client,
                model_name=model_name,
            )
            scores.append(score)
        except Exception as e:
            emit(f"[ERROR] {task_name}: {e}")
            emit_start(task_name)
            emit_end(task_name, 0.1, 0)
            scores.append(0.1)

    mean_score = sum(scores) / len(scores) if scores else 0.0
    emit(f"[SUMMARY] mean_score={mean_score:.4f} scores={scores}")


if __name__ == "__main__":
    main()
