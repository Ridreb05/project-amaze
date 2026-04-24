"""
Pydantic Models
===============
Clean client/server interface models for spacecraft rendezvous environment.
Follows OpenEnv convention: action, observation, state as separate schemas.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
import numpy as np


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class RendezvousAction(BaseModel):
    """
    Thruster command issued by the LLM agent.
    Forces are in LVLH frame (Local Vertical Local Horizontal).
    """
    fx: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Radial thrust force (N). Positive = away from Earth.",
    )
    fy: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Along-track thrust force (N). Positive = prograde.",
    )
    reasoning: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Flight controller reasoning for this manoeuvre. Earns bonus reward.",
    )

    @field_validator("fx", "fy")
    @classmethod
    def clamp_thrust(cls, v: float) -> float:
        return float(np.clip(v, -2.0, 2.0))

    class Config:
        json_schema_extra = {
            "example": {
                "fx": -0.5,
                "fy": 0.8,
                "reasoning": "Applying prograde burn to reduce closing distance. "
                             "Conserving radial fuel for final approach correction.",
            }
        }


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class RendezvousObservation(BaseModel):
    """
    What the agent observes at each step.
    Position/velocity includes sensor noise when comms_active=True.
    All readings are zeroed during comms blackout.
    """
    # Noisy sensor readings (zeroed during blackout)
    x_m: float = Field(description="Estimated radial position (m). Noisy.")
    y_m: float = Field(description="Estimated along-track position (m). Noisy.")
    vx_ms: float = Field(description="Estimated radial velocity (m/s). Noisy.")
    vy_ms: float = Field(description="Estimated along-track velocity (m/s). Noisy.")

    # Resources
    fuel_kg: float = Field(description="Remaining propellant mass (kg).")
    fuel_pct: float = Field(description="Fuel remaining as % of initial budget.")

    # Navigation
    estimated_distance_m: float = Field(description="Estimated distance to target (m).")
    estimated_speed_ms: float = Field(description="Estimated approach speed (m/s).")
    los_angle_deg: float = Field(description="Approach angle relative to docking axis (deg).")
    los_violation: bool = Field(description="True if outside ±45° docking corridor at <50m.")

    # Episode state
    step: int = Field(description="Current step number.")
    max_steps: int = Field(description="Maximum steps before timeout.")
    steps_remaining: int = Field(description="Steps remaining.")
    comms_active: bool = Field(description="False during scheduled blackout windows.")

    # Outcome
    done: bool = Field(description="True if episode has ended.")
    docked: bool = Field(description="True if successful docking achieved.")
    mission_aborted: bool = Field(description="True if abort condition triggered.")

    # Reward
    step_reward: float = Field(description="Reward received this step.")
    cumulative_reward: float = Field(description="Total reward so far this episode.")

    # Context
    scenario_difficulty: str = Field(description="warm_start | easy | medium | hard")
    message: str = Field(description="Human-readable status for LLM context.")

    class Config:
        json_schema_extra = {
            "example": {
                "x_m": 12.3, "y_m": 45.1, "vx_ms": -0.3, "vy_ms": -0.8,
                "fuel_kg": 6.2, "fuel_pct": 77.5,
                "estimated_distance_m": 46.7, "estimated_speed_ms": 0.85,
                "los_angle_deg": 15.2, "los_violation": False,
                "step": 7, "max_steps": 80, "steps_remaining": 73,
                "comms_active": True, "done": False, "docked": False,
                "mission_aborted": False, "step_reward": 0.42,
                "cumulative_reward": 1.85, "scenario_difficulty": "medium",
                "message": "Approach phase: 46.7m to target. Fuel 77.5%. LOS nominal.",
            }
        }


# ---------------------------------------------------------------------------
# State (internal, returned by /state endpoint)
# ---------------------------------------------------------------------------

class RendezvousState(BaseModel):
    """Full internal state — includes ground truth (no noise)."""
    episode_id: str
    seed: int
    step: int
    max_steps: int
    # Ground truth position/velocity
    true_x: float
    true_y: float
    true_vx: float
    true_vy: float
    true_distance: float
    true_speed: float
    # Resources
    fuel_kg: float
    initial_fuel_kg: float
    fuel_consumed_kg: float
    # Status
    comms_active: bool
    blackout_steps_remaining: List[int]
    los_violation: bool
    docked: bool
    mission_aborted: bool
    done: bool
    # Reward
    cumulative_reward: float
    reward_breakdown: Dict[str, float]
    # Scenario
    difficulty: str
    scenario_description: str


# ---------------------------------------------------------------------------
# Reset / Step request bodies
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed: Optional[int] = Field(default=42, description="RNG seed for reproducibility.")
    difficulty: Optional[str] = Field(
        default=None,
        description="warm_start | easy | medium | hard | None (auto-curriculum)",
    )

    class Config:
        json_schema_extra = {"example": {"seed": 42, "difficulty": "medium"}}


class StepRequest(BaseModel):
    action: RendezvousAction

    class Config:
        json_schema_extra = {
            "example": {
                "action": {"fx": -0.5, "fy": 0.8,
                           "reasoning": "Decelerating for final approach."}
            }
        }


# ---------------------------------------------------------------------------
# Step response
# ---------------------------------------------------------------------------

class StepResponse(BaseModel):
    observation: RendezvousObservation
    reward: float
    done: bool
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# Health / schema
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "healthy"
    environment: str = "spacecraft-rendezvous"
    version: str = "1.0.0"
