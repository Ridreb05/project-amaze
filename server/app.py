"""
FastAPI Application — Spacecraft Rendezvous OpenEnv Environment
===============================================================
Exposes the standard OpenEnv interface as a REST API.
Deployed to HuggingFace Spaces on port 7860 via Docker.

Endpoints:
    POST /reset      — Start new episode
    POST /step       — Apply action, advance environment
    GET  /state      — Full internal state (ground truth)
    GET  /health     — Liveness probe
    GET  /schema     — Action + observation JSON schemas
    POST /grade      — Score completed episode
    GET  /baseline   — Run baseline greedy agent, return scores
    GET  /info       — Environment metadata
    GET  /           — Redirect to /health
"""

import os
import sys
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.rendezvous_environment import RendezvousEnvironment
from models import (
    RendezvousAction, RendezvousObservation, RendezvousState,
    ResetRequest, StepRequest, StepResponse, HealthResponse,
)

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spacecraft-rendezvous")

# Global environment instance (single-session for hackathon simplicity)
env = RendezvousEnvironment()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Spacecraft Rendezvous Environment starting...")
    logger.info("Real-world reference: APIARY/Astrobee ISS experiment (NRL, May 2025)")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Spacecraft Rendezvous — OpenEnv RL Environment",
    description=(
        "Train LLM agents to perform autonomous spacecraft proximity operations "
        "using CWH relative motion dynamics. Same physics model as NASA mission planning. "
        "Inspired by the APIARY experiment aboard the ISS (May 2025)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/health")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness check. Returns healthy when server is ready."""
    return HealthResponse(
        status="healthy",
        environment="spacecraft-rendezvous",
        version="1.0.0",
    )


@app.post("/reset", response_model=RendezvousObservation, tags=["Environment"])
async def reset(request: ResetRequest):
    """
    Start a new episode.

    Parameters:
    - seed: RNG seed for reproducibility (default 42)
    - difficulty: warm_start | easy | medium | hard | null (auto-curriculum)

    Returns first observation.
    """
    try:
        obs = env.reset(
            seed=request.seed if request.seed is not None else 42,
            difficulty=request.difficulty,
        )
        logger.info(
            f"Reset: seed={request.seed} difficulty={request.difficulty} "
            f"dist={obs.estimated_distance_m:.1f}m"
        )
        return obs
    except Exception as e:
        logger.error(f"Reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResponse, tags=["Environment"])
async def step(request: StepRequest):
    try:
        response = env.step(request.action)
        return response
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Step error: {tb}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)} | {tb}")
    """
    Apply one action and advance the environment.

    Body: { "action": { "fx": float, "fy": float, "reasoning": str } }

    Returns observation, reward, done flag, and diagnostic info.
    """
    try:
        response = env.step(request.action)
        return response
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Step error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=RendezvousState, tags=["Environment"])
async def state():
    """
    Return full internal environment state (ground truth, no sensor noise).
    Useful for debugging and training diagnostics.
    """
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/grade", tags=["Evaluation"])
async def grade():
    """
    Grade the current (completed or in-progress) episode.
    Returns score in (0.0, 1.0) plus full breakdown.
    """
    try:
        result = env.grade()
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/schema", tags=["System"])
async def schema():
    """Return JSON schemas for action and observation models."""
    return {
        "action_schema": RendezvousAction.model_json_schema(),
        "observation_schema": RendezvousObservation.model_json_schema(),
        "state_schema": RendezvousState.model_json_schema(),
        "example_action": {
            "fx": -0.5,
            "fy": 0.8,
            "reasoning": "Decelerating for terminal approach. Conserving radial fuel.",
        },
    }


@app.get("/baseline", tags=["Evaluation"])
async def baseline():
    """
    Run the greedy baseline agent on 3 representative scenarios.
    Returns scores and episode summaries.
    Useful for comparing trained vs untrained agent performance.
    """
    results = []
    scenarios = [
        {"seed": 42, "difficulty": "easy", "label": "Easy — 50m, no blackout"},
        {"seed": 123, "difficulty": "medium", "label": "Medium — 100m, 2 blackouts"},
        {"seed": 777, "difficulty": "hard", "label": "Hard — 200m, tight fuel"},
    ]

    for scenario in scenarios:
        obs = env.reset(seed=scenario["seed"], difficulty=scenario["difficulty"])
        total_reward = 0.0
        steps = 0

        while not obs.done:
            # Greedy baseline: always thrust toward target
            if obs.comms_active and obs.estimated_distance_m > 0:
                # Normalised direction vector
                mag = max(obs.estimated_distance_m, 1e-6)
                fx = -obs.x_m / mag * 1.0   # thrust toward origin
                fy = -obs.y_m / mag * 1.0
                # Scale down for close approach
                scale = min(1.0, obs.estimated_distance_m / 20.0)
                action = RendezvousAction(
                    fx=fx * scale,
                    fy=fy * scale,
                    reasoning="Greedy baseline: thrust toward target.",
                )
            else:
                # Blackout: hold position (zero thrust)
                action = RendezvousAction(
                    fx=0.0, fy=0.0,
                    reasoning="Blackout: maintaining current trajectory.",
                )

            resp = env.step(action)
            obs = resp.observation
            total_reward += resp.reward
            steps += 1

        grade_result = env.grade()
        results.append({
            "label": scenario["label"],
            "difficulty": scenario["difficulty"],
            "seed": scenario["seed"],
            "score": grade_result["score"],
            "passed": grade_result["passed"],
            "docked": grade_result["docked"],
            "steps": steps,
            "cumulative_reward": round(total_reward, 3),
            "fuel_remaining_pct": grade_result["fuel_efficiency_pct"],
            "final_distance_m": grade_result["final_distance_m"],
        })

    return {
        "agent": "greedy-baseline",
        "description": "Thrusts directly toward target. No planning, no fuel conservation.",
        "results": results,
        "mean_score": round(sum(r["score"] for r in results) / len(results), 4),
        "dock_rate": f"{sum(1 for r in results if r['docked'])}/{len(results)}",
    }


@app.get("/info", tags=["System"])
async def info():
    """Environment metadata and configuration."""
    return {
        "name": "spacecraft-rendezvous",
        "version": "1.0.0",
        "theme": "World Modeling — Professional Tasks (Theme 3.1)",
        "description": (
            "LLM agent learns autonomous spacecraft proximity operations — "
            "rendezvous and docking under fuel constraints, sensor noise, "
            "and communication blackouts. Same physics as NASA mission planning."
        ),
        "real_world_reference": [
            "APIARY/Astrobee ISS experiment (NRL, May 2025)",
            "InnoCube/LeLaR CubeSat in-orbit RL (January 2025)",
            "ESA ClearSpace-1 debris removal mission",
            "ISRO SPADEX docking demonstration",
        ],
        "physics_model": "Clohessy-Wiltshire-Hill (CWH) relative motion equations",
        "action_space": {
            "fx": "Radial thrust force [-2.0, 2.0] Newtons",
            "fy": "Along-track thrust force [-2.0, 2.0] Newtons",
            "reasoning": "Optional: flight controller reasoning text",
        },
        "observation_space": {
            "position": "Noisy LVLH relative position (metres)",
            "velocity": "Noisy relative velocity (m/s)",
            "fuel_kg": "Remaining propellant (kg)",
            "comms_active": "False during scheduled blackout windows",
            "los_violation": "True if outside ±45° docking corridor",
        },
        "reward_components": {
            "approach_progress": "Dense: +0 to +2 per step",
            "fuel_efficiency": "Penalty: -0.05 per kg consumed",
            "los_constraint": "Penalty: -0.5 per violation",
            "blackout_handling": "Bonus: +0.3 per step during blackout",
            "terminal_docking": "Sparse: +5 to +10 success, -2 failure",
        },
        "difficulty_levels": ["warm_start", "easy", "medium", "hard"],
        "training_recommendation": "GRPO via Unsloth on Qwen2.5-1.5B-Instruct",
        "port": 7860,
    }


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    port = int(os.environ.get("PORT", 7860))
    reload = os.environ.get("ENV", "production") == "development"
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
