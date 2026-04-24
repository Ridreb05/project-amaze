"""
Rendezvous Environment
======================
Core environment class implementing OpenEnv's Environment interface.
Wires CWH physics + scenario generator + reward calculator into
the standard reset() / step() / state() / grade() API.
"""

import uuid
import numpy as np
from typing import Optional, Tuple, Dict, Any

from simulation.cwh_dynamics import CWHDynamics, CWHState, ThrustAction
from simulation.scenario_generator import ScenarioGenerator, ScenarioConfig
from simulation.reward_calculator import RewardCalculator
from models import (
    RendezvousAction, RendezvousObservation, RendezvousState,
    StepResponse,
)


class RendezvousEnvironment:
    """
    Spacecraft Proximity Operations RL Environment.

    Implements the OpenEnv standard interface:
        reset(seed, difficulty) → RendezvousObservation
        step(action)           → StepResponse
        state()                → RendezvousState
        grade()                → dict (score in [0,1])

    Physics: Clohessy-Wiltshire-Hill (CWH) relative motion equations.
    Domain: Spacecraft rendezvous and docking in LEO.
    Real-world ref: APIARY/Astrobee ISS (NRL, May 2025), ISRO SPADEX.
    """

    def __init__(self):
        self.dynamics = CWHDynamics()
        self.scenario_gen = ScenarioGenerator()
        self.reward_calc = RewardCalculator()

        # Episode state
        self._episode_id: str = ""
        self._seed: int = 42
        self._scenario: Optional[ScenarioConfig] = None
        self._cwh_state: Optional[CWHState] = None
        self._rng: Optional[np.random.Generator] = None
        self._cumulative_reward: float = 0.0
        self._step_rewards: list = []
        self._initial_fuel: float = 0.0
        self._initial_distance: float = 0.0

    # ── Public API ─────────────────────────────────────────────────────────

    def reset(
        self,
        seed: int = 42,
        difficulty: Optional[str] = None,
    ) -> RendezvousObservation:
        """Start a fresh episode. Returns first observation."""
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._episode_id = str(uuid.uuid4())[:8]
        self._cumulative_reward = 0.0
        self._step_rewards = []
        self.reward_calc.reset()

        # Generate scenario
        self._scenario = self.scenario_gen.generate(seed=seed, difficulty=difficulty)
        self._cwh_state = self._scenario.to_cwh_state()
        self._initial_fuel = self._scenario.fuel_kg
        self._initial_distance = self._scenario.initial_distance

        return self._build_observation(step_reward=0.0)

    def step(self, action: RendezvousAction) -> StepResponse:
        """Apply one action, advance physics, compute reward."""
        if self._cwh_state is None:
            raise RuntimeError("Must call reset() before step().")
        if self._cwh_state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        thrust = ThrustAction(fx=action.fx, fy=action.fy)
        prev_state = self._cwh_state

        # Update comms status before stepping
        current_step = self._cwh_state.step
        comms_active = current_step not in self._scenario.blackout_steps
        self._cwh_state.comms_active = comms_active

        # Propagate physics
        new_state, done, info = self.dynamics.step(
            state=self._cwh_state,
            action=thrust,
            rng=self._rng,
        )

        # Propagate comms to new state
        next_step = new_state.step
        new_state.comms_active = next_step not in self._scenario.blackout_steps

        self._cwh_state = new_state

        # Compute step reward
        step_rb = self.reward_calc.step_reward(
            prev_state=prev_state,
            new_state=new_state,
            action=thrust,
            info=info,
            reasoning_text=action.reasoning,
        )

        # Terminal reward if done
        terminal_rb = None
        if done:
            terminal_rb = self.reward_calc.terminal_reward(
                final_state=new_state,
                info=info,
            )

        total_step_reward = step_rb.total + (terminal_rb.total if terminal_rb else 0.0)
        self._cumulative_reward += total_step_reward
        self._step_rewards.append(total_step_reward)

        obs = self._build_observation(step_reward=total_step_reward)

        full_info = {
            **info,
            **step_rb.to_dict(),
            "comms_active": comms_active,
            "cumulative_reward": round(self._cumulative_reward, 4),
        }
        if terminal_rb:
            full_info["terminal_reward_breakdown"] = terminal_rb.to_dict()

        return StepResponse(
            observation=obs,
            reward=round(total_step_reward, 4),
            done=done,
            info=full_info,
        )

    def state(self) -> RendezvousState:
        """Return full internal state (ground truth, no noise)."""
        s = self._cwh_state
        if s is None:
            raise RuntimeError("Must call reset() first.")

        reward_summary = self.reward_calc.episode_summary()
        remaining_blackouts = [
            step for step in self._scenario.blackout_steps
            if step > s.step
        ]

        return RendezvousState(
            episode_id=self._episode_id,
            seed=self._seed,
            step=s.step,
            max_steps=s.max_steps,
            true_x=round(s.x, 4),
            true_y=round(s.y, 4),
            true_vx=round(s.vx, 4),
            true_vy=round(s.vy, 4),
            true_distance=round(s.distance, 4),
            true_speed=round(s.speed, 4),
            fuel_kg=round(s.fuel_kg, 4),
            initial_fuel_kg=self._initial_fuel,
            fuel_consumed_kg=round(self._initial_fuel - s.fuel_kg, 4),
            comms_active=s.comms_active,
            blackout_steps_remaining=remaining_blackouts,
            los_violation=s.los_violation,
            docked=s.docked,
            mission_aborted=s.mission_aborted,
            done=s.done,
            cumulative_reward=round(self._cumulative_reward, 4),
            reward_breakdown=reward_summary,
            difficulty=self._scenario.difficulty,
            scenario_description=self._scenario.description,
        )

    def grade(self) -> Dict[str, Any]:
        """
        Grade completed episode.
        Returns score in (0.0, 1.0) plus detailed breakdown.
        """
        if self._cwh_state is None:
            return {"score": 0.1, "passed": False, "reason": "No episode run."}

        s = self._cwh_state
        summary = self.reward_calc.episode_summary()
        raw_reward = self._cumulative_reward

        # Normalise raw reward to [0,1] score
        # Episode range: approx −3 (worst crash) to +10 (perfect docking)
        score = (raw_reward - (-3.0)) / (10.0 - (-3.0))
        score = float(np.clip(score, 0.02, 0.98))  # strict open interval

        # Determine pass/fail
        passed = s.docked and score > 0.5

        reason = []
        if s.docked:
            reason.append(f"Successful docking in {s.step} steps.")
            reason.append(f"Fuel remaining: {s.fuel_kg:.2f}kg ({100*s.fuel_kg/self._initial_fuel:.0f}%).")
        elif s.mission_aborted:
            reason.append("Mission abort: collision, out-of-bounds, or fuel exhaustion.")
            reason.append(f"Final distance: {s.distance:.1f}m.")
        else:
            reason.append(f"Timeout after {s.step} steps.")
            reason.append(f"Final distance: {s.distance:.1f}m.")

        return {
            "score": round(score, 4),
            "passed": passed,
            "docked": s.docked,
            "steps_taken": s.step,
            "fuel_remaining_kg": round(s.fuel_kg, 4),
            "fuel_efficiency_pct": round(100 * s.fuel_kg / max(self._initial_fuel, 1e-6), 1),
            "final_distance_m": round(s.distance, 4),
            "final_speed_ms": round(s.speed, 4),
            "cumulative_reward": round(raw_reward, 4),
            "reward_breakdown": summary,
            "reason": " ".join(reason),
            "difficulty": self._scenario.difficulty,
        }

    # ── Private helpers ────────────────────────────────────────────────────

    def _build_observation(self, step_reward: float) -> RendezvousObservation:
        """Build the agent-facing observation from current state."""
        s = self._cwh_state

        if s.comms_active:
            noisy = s.noisy_observation(self._rng)
            x_m = round(noisy.x, 3)
            y_m = round(noisy.y, 3)
            vx_ms = round(noisy.vx, 4)
            vy_ms = round(noisy.vy, 4)
            est_dist = round(noisy.distance, 2)
            est_speed = round(noisy.speed, 4)
        else:
            # Blackout: zero out all sensor readings
            x_m = y_m = vx_ms = vy_ms = 0.0
            est_dist = 0.0
            est_speed = 0.0

        fuel_pct = round(100.0 * s.fuel_kg / max(self._initial_fuel, 1e-6), 1)

        # Build human-readable message for LLM prompt context
        message = self._build_message(s, est_dist, fuel_pct)

        return RendezvousObservation(
            x_m=x_m, y_m=y_m, vx_ms=vx_ms, vy_ms=vy_ms,
            fuel_kg=round(s.fuel_kg, 3),
            fuel_pct=fuel_pct,
            estimated_distance_m=est_dist,
            estimated_speed_ms=est_speed,
            los_angle_deg=round(s.los_angle_deg, 1),
            los_violation=s.los_violation,
            step=s.step,
            max_steps=s.max_steps,
            steps_remaining=s.max_steps - s.step,
            comms_active=s.comms_active,
            done=s.done,
            docked=s.docked,
            mission_aborted=s.mission_aborted,
            step_reward=round(step_reward, 4),
            cumulative_reward=round(self._cumulative_reward, 4),
            scenario_difficulty=self._scenario.difficulty,
            message=message,
        )

    def _build_message(
        self,
        s: CWHState,
        est_dist: float,
        fuel_pct: float,
    ) -> str:
        """
        Human-readable status message injected into the LLM prompt.
        Designed to give the model maximum useful context in plain English.
        """
        if s.docked:
            return f"DOCKING SUCCESSFUL after {s.step} steps. Fuel remaining: {fuel_pct:.0f}%."

        if s.mission_aborted:
            return f"MISSION ABORT at step {s.step}. Distance: {est_dist:.1f}m."

        if not s.comms_active:
            return (
                f"COMMS BLACKOUT — step {s.step}/{s.max_steps}. "
                f"All sensors offline. Last known distance unavailable. "
                f"Fuel: {fuel_pct:.0f}%. Maintain conservative trajectory."
            )

        phase = "Terminal" if est_dist < 10 else "Final" if est_dist < 30 else "Mid" if est_dist < 80 else "Far"
        los_msg = " ⚠ LOS VIOLATION — adjust approach angle!" if s.los_violation else " LOS nominal."
        fuel_warn = " ⚠ LOW FUEL!" if fuel_pct < 25 else ""

        return (
            f"{phase} approach phase — step {s.step}/{s.max_steps}. "
            f"Distance: {est_dist:.1f}m. Speed: {s.speed:.3f}m/s. "
            f"Fuel: {fuel_pct:.0f}%.{los_msg}{fuel_warn}"
        )
