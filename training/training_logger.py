"""
Training Logger
===============
Saves every GRPO training episode to training_log.json.
Used by the demo/index.html replay dashboard to show agent
improvement visually without requiring live model inference.
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class StepLog:
    step: int
    x_m: float
    y_m: float
    distance_m: float
    fuel_pct: float
    comms_active: bool
    reward: float
    fx: float
    fy: float
    reasoning: str
    los_violation: bool


@dataclass
class EpisodeLog:
    episode: int
    seed: int
    difficulty: str
    docked: bool
    total_reward: float
    steps_taken: int
    final_distance_m: float
    fuel_remaining_pct: float
    initial_distance_m: float
    trajectory: List[Dict] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class TrainingLogger:
    """
    Logs training episodes incrementally to disk.
    Append-safe: new episodes are added to existing log file.
    """

    def __init__(self, log_path: str = "training_log.json"):
        self.log_path = log_path
        self._episodes: List[Dict] = []
        self._current_episode: Optional[EpisodeLog] = None
        self._reward_curve: List[float] = []

        # Load existing log if present
        if os.path.exists(log_path):
            try:
                with open(log_path, "r") as f:
                    data = json.load(f)
                    self._episodes = data.get("episodes", [])
                    self._reward_curve = data.get("reward_curve", [])
            except (json.JSONDecodeError, KeyError):
                pass

    def start_episode(
        self,
        episode: int,
        seed: int,
        difficulty: str,
        initial_distance: float,
    ) -> None:
        self._current_episode = EpisodeLog(
            episode=episode,
            seed=seed,
            difficulty=difficulty,
            docked=False,
            total_reward=0.0,
            steps_taken=0,
            final_distance_m=initial_distance,
            fuel_remaining_pct=100.0,
            initial_distance_m=initial_distance,
        )

    def log_step(
        self,
        step: int,
        obs: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
    ) -> None:
        if self._current_episode is None:
            return

        step_log = StepLog(
            step=step,
            x_m=round(obs.get("x_m", 0.0), 2),
            y_m=round(obs.get("y_m", 0.0), 2),
            distance_m=round(obs.get("estimated_distance_m", 0.0), 2),
            fuel_pct=round(obs.get("fuel_pct", 100.0), 1),
            comms_active=obs.get("comms_active", True),
            reward=round(reward, 4),
            fx=round(action.get("fx", 0.0), 3),
            fy=round(action.get("fy", 0.0), 3),
            reasoning=action.get("reasoning", "")[:200],
            los_violation=obs.get("los_violation", False),
        )
        self._current_episode.trajectory.append(asdict(step_log))
        self._current_episode.total_reward += reward
        self._current_episode.steps_taken = step

    def end_episode(
        self,
        docked: bool,
        final_distance_m: float,
        fuel_remaining_pct: float,
    ) -> None:
        if self._current_episode is None:
            return

        self._current_episode.docked = docked
        self._current_episode.final_distance_m = round(final_distance_m, 2)
        self._current_episode.fuel_remaining_pct = round(fuel_remaining_pct, 1)
        self._current_episode.total_reward = round(self._current_episode.total_reward, 4)

        self._episodes.append(asdict(self._current_episode))
        self._reward_curve.append(self._current_episode.total_reward)
        self._current_episode = None

        # Save after every episode
        self._save()

    def _save(self) -> None:
        """Save full log to disk. Safe to call frequently."""
        data = {
            "episodes": self._episodes,
            "reward_curve": self._reward_curve,
            "total_episodes": len(self._episodes),
            "dock_rate": (
                sum(1 for e in self._episodes if e.get("docked", False)) /
                max(len(self._episodes), 1)
            ),
            "mean_reward_last_20": (
                sum(self._reward_curve[-20:]) /
                max(len(self._reward_curve[-20:]), 1)
            ),
        }
        with open(self.log_path, "w") as f:
            json.dump(data, f, indent=2)

    def summary(self) -> Dict[str, Any]:
        n = len(self._episodes)
        if n == 0:
            return {"total_episodes": 0}

        dock_rate = sum(1 for e in self._episodes if e.get("docked")) / n
        rewards = self._reward_curve
        return {
            "total_episodes": n,
            "dock_rate": round(dock_rate, 3),
            "mean_reward_all": round(sum(rewards) / n, 4),
            "mean_reward_last_20": round(sum(rewards[-20:]) / max(len(rewards[-20:]), 1), 4),
            "best_reward": round(max(rewards), 4),
            "worst_reward": round(min(rewards), 4),
        }
