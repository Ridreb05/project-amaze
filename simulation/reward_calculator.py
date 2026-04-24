"""
Reward Calculator
=================
5-component reward function for spacecraft proximity operations.

Design principles (from hackathon self-serve guide):
1. Rich informative signal at every step — not just 0/1 at episode end
2. Multiple independent reward functions — harder to hack
3. Wide spread between good (+4 to +8) and bad (−2) episodes for GRPO
4. Hard to game — agent cannot get high reward without solving the real task

Component summary:
  R_approach      +0.0 to +2.0/step   Dense: proportional to distance reduction
  R_fuel          −0.05/unit thrust    Penalise wasteful burns
  R_los           −0.5 if violated     LOS cone constraint (±45° at <50m)
  R_blackout      +0.3 bonus/step      Sensible decisions during comms blackout
  R_terminal      +4 to +8 or −2      Sparse: docking success or failure
  R_reasoning     +0.1 if present      Bonus for structured reasoning text
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from simulation.cwh_dynamics import CWHState, ThrustAction, DOCKING_DISTANCE_M


# ---------------------------------------------------------------------------
# Reward weights — tune these for GRPO convergence
# ---------------------------------------------------------------------------
W_APPROACH: float = 2.0          # max per-step approach reward
W_FUEL_PENALTY: float = 0.05     # per kg of fuel consumed
W_LOS_PENALTY: float = 0.5       # per LOS violation
W_BLACKOUT_BONUS: float = 0.3    # per step during blackout (if reasonable action)
W_TERMINAL_SUCCESS: float = 5.0  # base terminal success reward
W_TERMINAL_BONUS_FUEL: float = 3.0  # bonus for fuel conservation at docking
W_TERMINAL_BONUS_SPEED: float = 2.0  # bonus for gentle approach speed
W_TERMINAL_FAILURE: float = -2.0    # timeout / crash / abort
W_FORMAT_BONUS: float = 0.1      # agent provided structured reasoning

# Bounds for GRPO — must have spread
REWARD_EPISODE_MIN: float = -3.0
REWARD_EPISODE_MAX: float = 10.0


@dataclass
class RewardBreakdown:
    """Detailed reward decomposition — logged during training for diagnostics."""
    approach: float = 0.0
    fuel: float = 0.0
    los: float = 0.0
    blackout: float = 0.0
    terminal: float = 0.0
    reasoning: float = 0.0

    @property
    def total(self) -> float:
        return (self.approach + self.fuel + self.los +
                self.blackout + self.terminal + self.reasoning)

    def to_dict(self) -> dict:
        return {
            "r_approach": round(self.approach, 4),
            "r_fuel": round(self.fuel, 4),
            "r_los": round(self.los, 4),
            "r_blackout": round(self.blackout, 4),
            "r_terminal": round(self.terminal, 4),
            "r_reasoning": round(self.reasoning, 4),
            "r_total": round(self.total, 4),
        }


class RewardCalculator:
    """
    Computes step and episode rewards for spacecraft proximity operations.

    GRPO needs variance. This calculator is designed to produce:
    - Warm-start easy scenarios:  +1.5 to +5.0 range
    - Standard successful episode: +4.0 to +8.0
    - Failed episode (timeout):   −2.0 to +0.5
    - Crashed episode:            −2.0 to −1.0
    """

    def __init__(self):
        self._episode_cumulative = RewardBreakdown()

    def reset(self):
        """Call at the start of each episode."""
        self._episode_cumulative = RewardBreakdown()

    def step_reward(
        self,
        prev_state: CWHState,
        new_state: CWHState,
        action: ThrustAction,
        info: dict,
        reasoning_text: Optional[str] = None,
    ) -> RewardBreakdown:
        """
        Compute step-level reward components.
        Called after every environment step.
        """
        rb = RewardBreakdown()

        # ── Component 1: Approach Progress ────────────────────────
        # Reward proportional to fractional distance reduction
        # Normalised so maximum per-step reward is W_APPROACH
        prev_dist = prev_state.distance
        curr_dist = new_state.distance
        if prev_dist > 1e-6:
            distance_reduction = (prev_dist - curr_dist) / prev_dist
            # Clip: no reward for moving away, max reward for rapid approach
            rb.approach = float(np.clip(distance_reduction * W_APPROACH, -0.3, W_APPROACH))
        else:
            rb.approach = 0.0

        # ── Component 2: Fuel Efficiency ──────────────────────────
        # Penalise fuel consumption per step
        fuel_consumed = info.get("fuel_consumed_kg", 0.0)
        rb.fuel = -W_FUEL_PENALTY * fuel_consumed * 10.0  # scale to sensible range

        # ── Component 3: LOS Constraint ───────────────────────────
        if info.get("los_violation", False):
            rb.los = -W_LOS_PENALTY

        # ── Component 4: Comms Blackout Handling ──────────────────
        # Bonus if agent makes a reasonable decision during blackout
        # "Reasonable" = small thrust magnitude (not panicking) or holding course
        if not prev_state.comms_active:
            thrust_mag = action.magnitude
            if thrust_mag < 0.5:  # conservative action during blackout = good
                rb.blackout = W_BLACKOUT_BONUS
            elif thrust_mag < 1.0:
                rb.blackout = W_BLACKOUT_BONUS * 0.5
            # Large thrust during blackout = no bonus (blind firing)

        # ── Component 6: Reasoning Quality ────────────────────────
        if reasoning_text and len(reasoning_text.strip()) > 20:
            rb.reasoning = W_FORMAT_BONUS

        # Accumulate
        self._episode_cumulative.approach += rb.approach
        self._episode_cumulative.fuel += rb.fuel
        self._episode_cumulative.los += rb.los
        self._episode_cumulative.blackout += rb.blackout
        self._episode_cumulative.reasoning += rb.reasoning

        return rb

    def terminal_reward(
        self,
        final_state: CWHState,
        info: dict,
    ) -> RewardBreakdown:
        """
        Compute terminal reward at episode end.
        This is the biggest signal — dominates GRPO advantage estimation.
        """
        rb = RewardBreakdown()

        if final_state.docked:
            # Base docking reward
            rb.terminal = W_TERMINAL_SUCCESS

            # Bonus for fuel conservation (proportional to remaining fuel)
            fuel_ratio = final_state.fuel_kg / 10.0  # normalise to initial budget assumption
            rb.terminal += W_TERMINAL_BONUS_FUEL * min(fuel_ratio, 1.0)

            # Bonus for gentle closing velocity at docking
            speed = final_state.speed
            if speed < 0.02:
                rb.terminal += W_TERMINAL_BONUS_SPEED  # perfect approach
            elif speed < 0.05:
                rb.terminal += W_TERMINAL_BONUS_SPEED * 0.5  # good approach
            # Above DOCKING_VELOCITY_MS threshold: no speed bonus

        elif final_state.mission_aborted:
            # Crash / out of bounds / fuel exhausted far from target
            rb.terminal = W_TERMINAL_FAILURE

            # Partial credit if they got close before failing
            dist = final_state.distance
            if dist < 10.0:
                rb.terminal += 0.5  # got close, then something went wrong
            elif dist < 30.0:
                rb.terminal += 0.2

        else:
            # Timeout — ran out of steps
            rb.terminal = W_TERMINAL_FAILURE

            # Partial credit based on final distance
            dist = final_state.distance
            initial_dist = final_state.prev_distance  # best proxy for initial
            if dist < 2.0:
                rb.terminal += 1.8  # basically docked, just didn't meet velocity
            elif dist < 10.0:
                rb.terminal += 1.2
            elif dist < 30.0:
                rb.terminal += 0.6
            elif dist < 50.0:
                rb.terminal += 0.2

        self._episode_cumulative.terminal += rb.terminal
        return rb

    def episode_summary(self) -> dict:
        """Return full episode reward breakdown for logging."""
        return {
            **self._episode_cumulative.to_dict(),
            "r_episode_total": round(self._episode_cumulative.total, 4),
        }

    @staticmethod
    def normalize_for_grpo(reward: float) -> float:
        """
        Normalize episode reward to a range suitable for GRPO.
        GRPO works best when rewards are in roughly [-3, +10] range.
        Clamp to prevent outlier episodes from dominating gradients.
        """
        return float(np.clip(reward, REWARD_EPISODE_MIN, REWARD_EPISODE_MAX))
