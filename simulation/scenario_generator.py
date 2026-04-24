"""
Scenario Generator
==================
Generates reproducible spacecraft proximity operations scenarios.
Includes warm-start curriculum: easy scenarios first to ensure non-zero
reward during early GRPO training episodes.

Design principle (from hackathon self-serve guide):
"The task must be hard enough to be interesting, but not so hard that
the model never succeeds — RL only works if P(good answer) > 0."
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from simulation.cwh_dynamics import CWHState, CHASER_MASS_KG


@dataclass
class ScenarioConfig:
    """Complete scenario specification."""
    # Initial relative position (metres)
    x0: float
    y0: float
    # Initial relative velocity (m/s)
    vx0: float
    vy0: float
    # Resources
    fuel_kg: float
    # Difficulty modifiers
    sensor_noise_std: float         # position noise std dev (metres)
    blackout_steps: List[int]       # step indices where comms cut out
    max_steps: int
    # Metadata
    difficulty: str                 # "warm_start" | "easy" | "medium" | "hard"
    seed: int
    description: str = ""

    @property
    def initial_distance(self) -> float:
        return float(np.sqrt(self.x0 ** 2 + self.y0 ** 2))

    def to_cwh_state(self) -> CWHState:
        return CWHState(
            x=self.x0,
            y=self.y0,
            vx=self.vx0,
            vy=self.vy0,
            fuel_kg=self.fuel_kg,
            step=0,
            max_steps=self.max_steps,
            comms_active=True,
            sensor_noise_std=self.sensor_noise_std,
            prev_distance=float(np.sqrt(self.x0 ** 2 + self.y0 ** 2)),
            mission_aborted=False,
            docked=False,
        )


class ScenarioGenerator:
    """
    Generates reproducible scenarios with curriculum difficulty control.

    Curriculum levels:
    - warm_start: 20m distance, no noise, no blackout, abundant fuel
                  Ensures GRPO gets at least SOME positive signal early
    - easy:       50m distance, low noise, no blackout, generous fuel
    - medium:     100m distance, moderate noise, 2 blackout windows, normal fuel
    - hard:       200m+ distance, high noise, 3+ blackout windows, tight fuel
    """

    # Difficulty distribution for GRPO training
    # More warm/easy early → curriculum shifts to harder as training progresses
    DIFFICULTY_WEIGHTS = {
        "warm_start": 0.15,
        "easy": 0.35,
        "medium": 0.35,
        "hard": 0.15,
    }

    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed

    def generate(self, seed: int, difficulty: Optional[str] = None) -> ScenarioConfig:
        """
        Generate a scenario. If difficulty is None, sample from curriculum distribution.
        """
        rng = np.random.default_rng(seed)

        if difficulty is None:
            levels = list(self.DIFFICULTY_WEIGHTS.keys())
            weights = list(self.DIFFICULTY_WEIGHTS.values())
            difficulty = rng.choice(levels, p=weights)

        if difficulty == "warm_start":
            return self._warm_start(seed, rng)
        elif difficulty == "easy":
            return self._easy(seed, rng)
        elif difficulty == "medium":
            return self._medium(seed, rng)
        elif difficulty == "hard":
            return self._hard(seed, rng)
        else:
            raise ValueError(f"Unknown difficulty: {difficulty}")

    def _warm_start(self, seed: int, rng: np.random.Generator) -> ScenarioConfig:
        """
        Very forgiving scenario — agent almost cannot fail.
        Initial distance 15-25m, huge fuel budget, no noise, no blackout.
        Ensures early GRPO episodes produce non-zero reward.
        """
        # Place chaser directly ahead (along-track) — most natural approach geometry
        y0 = rng.uniform(15.0, 25.0)
        x0 = rng.uniform(-3.0, 3.0)  # slight radial offset
        vx0 = rng.uniform(-0.05, 0.05)
        vy0 = rng.uniform(-0.2, -0.05)  # slight closing velocity

        return ScenarioConfig(
            x0=float(x0), y0=float(y0),
            vx0=float(vx0), vy0=float(vy0),
            fuel_kg=20.0,           # very generous — ~40 full burns
            sensor_noise_std=0.05,  # near-perfect sensors
            blackout_steps=[],
            max_steps=40,
            difficulty="warm_start",
            seed=seed,
            description=f"Warm-start: {y0:.1f}m ahead, generous fuel, perfect sensors",
        )

    def _easy(self, seed: int, rng: np.random.Generator) -> ScenarioConfig:
        """40-60m initial distance, low noise, no blackout, generous fuel."""
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(40.0, 60.0)
        x0 = dist * np.cos(angle)
        y0 = dist * np.sin(angle)

        # Small initial closing velocity
        vx0 = rng.uniform(-0.1, 0.1)
        vy0 = rng.uniform(-0.1, 0.1)

        return ScenarioConfig(
            x0=float(x0), y0=float(y0),
            vx0=float(vx0), vy0=float(vy0),
            fuel_kg=12.0,
            sensor_noise_std=0.3,
            blackout_steps=[],
            max_steps=60,
            difficulty="easy",
            seed=seed,
            description=f"Easy: {dist:.1f}m, low noise, no blackout",
        )

    def _medium(self, seed: int, rng: np.random.Generator) -> ScenarioConfig:
        """80-120m initial distance, moderate noise, 2 blackout windows."""
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(80.0, 120.0)
        x0 = dist * np.cos(angle)
        y0 = dist * np.sin(angle)

        vx0 = rng.uniform(-0.2, 0.2)
        vy0 = rng.uniform(-0.2, 0.2)

        # 2 blackout windows of 3 steps each
        max_steps = 80
        blackout_starts = sorted(rng.choice(range(15, max_steps - 10), size=2, replace=False))
        blackout_steps = []
        for bs in blackout_starts:
            blackout_steps.extend(range(bs, min(bs + 3, max_steps)))

        return ScenarioConfig(
            x0=float(x0), y0=float(y0),
            vx0=float(vx0), vy0=float(vy0),
            fuel_kg=8.0,
            sensor_noise_std=0.8,
            blackout_steps=list(set(blackout_steps)),
            max_steps=max_steps,
            difficulty="medium",
            seed=seed,
            description=f"Medium: {dist:.1f}m, noisy sensors, 2 blackout windows",
        )

    def _hard(self, seed: int, rng: np.random.Generator) -> ScenarioConfig:
        """150-300m distance, high noise, 3+ blackout windows, tight fuel."""
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(150.0, 300.0)
        x0 = dist * np.cos(angle)
        y0 = dist * np.sin(angle)

        # Nonzero initial velocity — adds trajectory complexity
        vx0 = rng.uniform(-0.5, 0.5)
        vy0 = rng.uniform(-0.5, 0.5)

        # 3-4 blackout windows
        max_steps = 100
        n_blackouts = rng.integers(3, 5)
        blackout_starts = sorted(rng.choice(range(10, max_steps - 15), size=n_blackouts, replace=False))
        blackout_steps = []
        for bs in blackout_starts:
            length = rng.integers(3, 6)
            blackout_steps.extend(range(bs, min(bs + length, max_steps)))

        return ScenarioConfig(
            x0=float(x0), y0=float(y0),
            vx0=float(vx0), vy0=float(vy0),
            fuel_kg=6.0,            # tight budget
            sensor_noise_std=1.5,
            blackout_steps=list(set(blackout_steps)),
            max_steps=max_steps,
            difficulty="hard",
            seed=seed,
            description=f"Hard: {dist:.1f}m, high noise, {n_blackouts} blackout windows, tight fuel",
        )

    def generate_training_batch(
        self,
        batch_size: int,
        episode_number: int,
        total_episodes: int,
    ) -> List[ScenarioConfig]:
        """
        Generate a batch of scenarios with adaptive curriculum.
        Early episodes: more warm_start/easy.
        Later episodes: more medium/hard.
        """
        progress = episode_number / max(total_episodes, 1)

        # Adaptive weights: shift difficulty as training progresses
        if progress < 0.2:
            weights = {"warm_start": 0.40, "easy": 0.40, "medium": 0.15, "hard": 0.05}
        elif progress < 0.5:
            weights = {"warm_start": 0.15, "easy": 0.40, "medium": 0.35, "hard": 0.10}
        elif progress < 0.8:
            weights = {"warm_start": 0.05, "easy": 0.25, "medium": 0.45, "hard": 0.25}
        else:
            weights = {"warm_start": 0.00, "easy": 0.15, "medium": 0.45, "hard": 0.40}

        scenarios = []
        rng = np.random.default_rng(self.base_seed + episode_number)
        levels = list(weights.keys())
        probs = list(weights.values())

        for i in range(batch_size):
            difficulty = rng.choice(levels, p=probs)
            seed = self.base_seed + episode_number * 1000 + i
            scenarios.append(self.generate(seed=seed, difficulty=difficulty))

        return scenarios
