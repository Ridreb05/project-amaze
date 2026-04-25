"""
FastAPI Application — Spacecraft Rendezvous OpenEnv Environment
===============================================================
FIX: HF Spaces load-balances across replicas — force single worker
and use session-based environment store so reset/step always hit
the same env instance.
"""

import os
import sys
import logging
import traceback
from contextlib import asynccontextmanager
from typing import Optional, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

# Force single worker — critical for stateful environment on HF Spaces
os.environ["WEB_CONCURRENCY"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.rendezvous_environment import RendezvousEnvironment
from models import (
    RendezvousAction, RendezvousObservation, RendezvousState,
    ResetRequest, StepRequest, StepResponse, HealthResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spacecraft-rendezvous")

# Session store — each client gets its own env instance
DEFAULT_SESSION = "default"
_sessions: Dict[str, RendezvousEnvironment] = {
    DEFAULT_SESSION: RendezvousEnvironment()
}


def get_env(session_id: Optional[str] = None) -> RendezvousEnvironment:
    sid = session_id or DEFAULT_SESSION
    if sid not in _sessions:
        logger.info(f"New session: {sid}")
        _sessions[sid] = RendezvousEnvironment()
    return _sessions[sid]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Spacecraft Rendezvous Environment starting — CWH physics, APIARY/ISS reference")
    yield
    logger.info("Shutdown.")


app = FastAPI(
    title="Spacecraft Rendezvous — OpenEnv RL Environment",
    description="Train LLM agents for autonomous spacecraft proximity operations using CWH dynamics.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/health")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(status="healthy", environment="spacecraft-rendezvous", version="1.0.0")


@app.post("/reset", response_model=RendezvousObservation, tags=["Environment"])
async def reset(request: ResetRequest, x_session_id: Optional[str] = Header(default=None)):
    try:
        env = get_env(x_session_id)
        obs = env.reset(
            seed=request.seed if request.seed is not None else 42,
            difficulty=request.difficulty,
        )
        logger.info(f"[{x_session_id or DEFAULT_SESSION}] Reset seed={request.seed} diff={request.difficulty} dist={obs.estimated_distance_m:.1f}m")
        return obs
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Reset error:\n{tb}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}\n\n{tb}")


@app.post("/step", response_model=StepResponse, tags=["Environment"])
async def step(request: StepRequest, x_session_id: Optional[str] = Header(default=None)):
    try:
        env = get_env(x_session_id)
        response = env.step(request.action)
        return response
    except RuntimeError as e:
        tb = traceback.format_exc()
        logger.error(f"Step RuntimeError:\n{tb}")
        raise HTTPException(status_code=400, detail=f"{str(e)}\n\n{tb}")
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Step error:\n{tb}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}\n\n{tb}")


@app.get("/state", response_model=RendezvousState, tags=["Environment"])
async def state(x_session_id: Optional[str] = Header(default=None)):
    try:
        return get_env(x_session_id).state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}\n\n{tb}")


@app.post("/grade", tags=["Evaluation"])
async def grade(x_session_id: Optional[str] = Header(default=None)):
    try:
        result = get_env(x_session_id).grade()
        return JSONResponse(content=result)
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}\n\n{tb}")


@app.get("/schema", tags=["System"])
async def schema():
    return {
        "action_schema": RendezvousAction.model_json_schema(),
        "observation_schema": RendezvousObservation.model_json_schema(),
        "state_schema": RendezvousState.model_json_schema(),
        "example_action": {"fx": -0.5, "fy": 0.8, "reasoning": "Decelerating for terminal approach."},
    }


@app.get("/baseline", tags=["Evaluation"])
async def baseline():
    baseline_env = RendezvousEnvironment()
    results = []
    for seed, diff, label in [
        (42, "easy", "Easy 50m no blackout"),
        (123, "medium", "Medium 100m 2 blackouts"),
        (777, "hard", "Hard 200m tight fuel"),
    ]:
        try:
            obs = baseline_env.reset(seed=seed, difficulty=diff)
            total_reward = 0.0
            steps = 0
            while not obs.done:
                if obs.comms_active and obs.estimated_distance_m > 0:
                    mag = max(obs.estimated_distance_m, 1e-6)
                    scale = min(1.0, obs.estimated_distance_m / 20.0)
                    action = RendezvousAction(
                        fx=(-obs.x_m / mag) * scale,
                        fy=(-obs.y_m / mag) * scale,
                        reasoning="Greedy: thrust toward target.",
                    )
                else:
                    action = RendezvousAction(fx=0.0, fy=0.0, reasoning="Blackout: holding.")
                resp = baseline_env.step(action)
                obs = resp.observation
                total_reward += resp.reward
                steps += 1
            g = baseline_env.grade()
            results.append({"label": label, "difficulty": diff, "seed": seed,
                           "score": g["score"], "docked": g["docked"], "steps": steps,
                           "cumulative_reward": round(total_reward, 3),
                           "fuel_remaining_pct": g["fuel_efficiency_pct"],
                           "final_distance_m": g["final_distance_m"]})
        except Exception as e:
            results.append({"label": label, "error": str(e)})

    return {
        "agent": "greedy-baseline",
        "results": results,
        "mean_score": round(sum(r.get("score", 0) for r in results) / max(len(results), 1), 4),
        "dock_rate": f"{sum(1 for r in results if r.get('docked'))}/{len(results)}",
    }


@app.get("/info", tags=["System"])
async def info():
    return {
        "name": "spacecraft-rendezvous",
        "version": "1.0.0",
        "theme": "World Modeling Professional Tasks Theme 3.1",
        "physics_model": "Clohessy-Wiltshire-Hill CWH relative motion equations",
        "real_world_reference": ["APIARY/Astrobee ISS (NRL May 2025)", "InnoCube/LeLaR CubeSat Jan 2025"],
        "difficulty_levels": ["warm_start", "easy", "medium", "hard"],
        "active_sessions": len(_sessions),
        "port": 7860,
    }


@app.get("/sessions", tags=["System"])
async def sessions():
    return {"active_sessions": list(_sessions.keys()), "count": len(_sessions)}


def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
