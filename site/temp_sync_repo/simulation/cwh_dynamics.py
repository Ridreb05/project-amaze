"""
Clohessy-Wiltshire-Hill (CWH) Relative Motion Dynamics
=======================================================
The CWH equations are the aerospace industry standard for modeling spacecraft
proximity operations. They describe relative motion in the Local Vertical Local
Horizontal (LVLH) frame of a circular reference orbit.

Reference: Clohessy, W.H. & Wiltshire, R.S. (1960). Terminal guidance system
for satellite rendezvous. Journal of the Aerospace Sciences, 27(9), 653-658.

Real-world usage: NASA APIARY (ISS, 2025), ESA ClearSpace-1, JAXA HTV docking.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Mean motion for LEO ~400km altitude (ISS orbit) — rad/s
# n = sqrt(mu / r^3), mu=3.986e14, r=6.771e6m → n ≈ 0.00113 rad/s
MEAN_MOTION_LEO: float = 0.00113          # rad/s
SIMULATION_DT: float = 10.0              # seconds per simulation step
DOCKING_DISTANCE_M: float = 0.5         # metres — success threshold
DOCKING_VELOCITY_MS: float = 0.05       # m/s — max closing speed at docking
MAX_THRUST_N: float = 2.0               # Newtons — single thruster max
CHASER_MASS_KG: float = 500.0           # kg — chaser spacecraft mass
ISP_S: float = 220.0                    # s — specific impulse (cold gas thruster)
G0_MS2: float = 9.80665                 # m/s^2 — standard gravity


@dataclass
class CWHState:
    """Full CWH relative state — chaser w.r.t. target in LVLH frame."""
    # Relative position (metres): x=radial, y=along-track, z=cross-track (2D sim → z=0)
    x: float = 0.0
    y: float = 0.0
    # Relative velocity (m/s)
    vx: float = 0.0
    vy: float = 0.0
    # Resources
    fuel_kg: float = 10.0
    # Episode meta
    step: int = 0
    max_steps: int = 100
    # Sensor / comms state
    comms_active: bool = True
    sensor_noise_std: float = 0.5       # metres — position noise std dev
    # History for reward shaping
    prev_distance: float = 0.0
    mission_aborted: bool = False
    docked: bool = False

    def __post_init__(self):
        self.prev_distance = self.distance

    @property
    def distance(self) -> float:
        """Euclidean distance to target (origin)."""
        return float(np.sqrt(self.x ** 2 + self.y ** 2))

    @property
    def speed(self) -> float:
        """Relative speed magnitude."""
        return float(np.sqrt(self.vx ** 2 + self.vy ** 2))

    @property
    def done(self) -> bool:
        return self.docked or self.mission_aborted or self.step >= self.max_steps

    @property
    def los_angle_deg(self) -> float:
        """Angle of approach relative to docking axis (+x radial). Degrees."""
        if self.distance < 1e-6:
            return 0.0
        # Vector from chaser to target
        dx, dy = -self.x, -self.y
        angle = float(np.degrees(np.arctan2(dy, dx)))
        return angle

    @property
    def los_violation(self) -> bool:
        """
        Line-of-sight (LOS) cone constraint.
        Agent must approach from within ±45° of the docking axis.
        Only enforced when within 50m — matches real KOZ (Keep-Out Zone) rules.
        """
        if self.distance > 50.0:
            return False
        angle = abs(self.los_angle_deg)
        # Normalise to [-180, 180]
        if angle > 180:
            angle = 360 - angle
        return angle > 45.0

    def noisy_observation(self, rng: np.random.Generator) -> "CWHState":
        """Return state copy with Gaussian sensor noise applied to position/velocity."""
        noisy = CWHState(
            x=self.x + rng.normal(0, self.sensor_noise_std),
            y=self.y + rng.normal(0, self.sensor_noise_std),
            vx=self.vx + rng.normal(0, self.sensor_noise_std * 0.1),
            vy=self.vy + rng.normal(0, self.sensor_noise_std * 0.1),
            fuel_kg=self.fuel_kg,
            step=self.step,
            max_steps=self.max_steps,
            comms_active=self.comms_active,
            sensor_noise_std=self.sensor_noise_std,
            prev_distance=self.prev_distance,
            mission_aborted=self.mission_aborted,
            docked=self.docked,
        )
        return noisy


@dataclass
class ThrustAction:
    """Thruster command — force in LVLH frame (Newtons)."""
    fx: float = 0.0     # radial thrust (N)
    fy: float = 0.0     # along-track thrust (N)

    def clamp(self) -> "ThrustAction":
        """Enforce actuator saturation limits."""
        return ThrustAction(
            fx=float(np.clip(self.fx, -MAX_THRUST_N, MAX_THRUST_N)),
            fy=float(np.clip(self.fy, -MAX_THRUST_N, MAX_THRUST_N)),
        )

    @property
    def magnitude(self) -> float:
        return float(np.sqrt(self.fx ** 2 + self.fy ** 2))

    @property
    def delta_v(self) -> float:
        """Impulsive delta-v (m/s) for this thrust over one DT step."""
        force = self.magnitude
        return (force / CHASER_MASS_KG) * SIMULATION_DT


class CWHDynamics:
    """
    Propagates CWH relative motion state one timestep forward.

    CWH equations (linearised, 2D in orbital plane):
        ẍ = 3n²x + 2nẏ + fx/m
        ÿ = −2nẋ   + fy/m

    where n = mean motion, x = radial, y = along-track.
    """

    def __init__(self, n: float = MEAN_MOTION_LEO, dt: float = SIMULATION_DT):
        self.n = n
        self.dt = dt
        self._build_stm()

    def _build_stm(self) -> None:
        """
        Build the State Transition Matrix (STM) for unforced CWH motion.
        This is the exact closed-form solution — no numerical integration error.
        Ref: Tschauner & Hempel (1965), exact CWH STM.
        """
        n, t = self.n, self.dt
        nt = n * t
        snt = np.sin(nt)
        cnt = np.cos(nt)

        # STM: maps [x, y, vx, vy] → [x', y', vx', vy'] (unforced)
        self.stm = np.array([
            [4 - 3 * cnt,        0,        snt / n,         2 * (1 - cnt) / n],
            [6 * (snt - nt),     1, -2 * (1 - cnt) / n, (4 * snt - 3 * nt) / n],
            [3 * n * snt,        0,        cnt,              2 * snt],
            [-6 * n * (1 - cnt), 0,       -2 * snt,         4 * cnt - 3],
        ])

        # Input matrix B: maps [fx/m, fy/m] → state delta (forced)
        # Derived from CWH particular solution
        self.B = np.array([
            [(1 - cnt) / n ** 2,         2 * (nt - snt) / n ** 2],
            [2 * (cnt - 1) / n ** 2 + (3 * nt ** 2) / (2 * n ** 2),
             (4 * snt - 3 * nt) / n ** 2 - 2 * (1 - cnt) / n ** 2],
            [snt / n,                    2 * (1 - cnt) / n],
            [-2 * (1 - cnt) / n,         (4 * cnt - 3)],
        ])

    def step(
        self,
        state: CWHState,
        action: ThrustAction,
        rng: np.random.Generator,
    ) -> Tuple[CWHState, float, bool, dict]:
        """
        Propagate state by one simulation timestep.

        Returns:
            new_state: updated CWHState
            reward: scalar reward for this step
            done: episode termination flag
            info: diagnostic dict
        """
        action = action.clamp()

        # ── Fuel consumption ───────────────────────────────────────
        dv = action.delta_v
        if dv > 0:
            # Tsiolkovsky: dm = m * (1 - exp(-dv / (Isp * g0)))
            dm = CHASER_MASS_KG * (1 - np.exp(-dv / (ISP_S * G0_MS2)))
            dm = min(dm, state.fuel_kg)  # cannot consume more than available
        else:
            dm = 0.0

        fuel_remaining = max(0.0, state.fuel_kg - dm)

        # ── CWH propagation ────────────────────────────────────────
        sv = np.array([state.x, state.y, state.vx, state.vy])

        if fuel_remaining > 0 or dm == 0:
            # Acceleration from thrust
            ax = action.fx / CHASER_MASS_KG
            ay = action.fy / CHASER_MASS_KG
            forced_delta = self.B @ np.array([ax, ay])
        else:
            # Out of fuel — coast
            forced_delta = np.zeros(4)

        sv_next = self.stm @ sv + forced_delta

        # ── Terminal condition checks ──────────────────────────────
        new_dist = float(np.sqrt(sv_next[0] ** 2 + sv_next[1] ** 2))
        new_speed = float(np.sqrt(sv_next[2] ** 2 + sv_next[3] ** 2))

        docked = (new_dist <= DOCKING_DISTANCE_M and
                  new_speed <= DOCKING_VELOCITY_MS and
                  state.step >= 3)  # minimum approach steps

        # Safety abort: out of bounds (> 2km) or collision (speed > 5 m/s at <2m)
        out_of_bounds = new_dist > 2000.0
        collision = (new_dist < 2.0 and new_speed > 5.0)
        mission_aborted = out_of_bounds or collision or (fuel_remaining <= 0 and new_dist > 5.0)

        # ── Comms blackout (scheduled per scenario) ────────────────
        # Blackout windows passed via state — handled at scenario level

        # ── Build new state ────────────────────────────────────────
        new_state = CWHState(
            x=float(sv_next[0]),
            y=float(sv_next[1]),
            vx=float(sv_next[2]),
            vy=float(sv_next[3]),
            fuel_kg=fuel_remaining,
            step=state.step + 1,
            max_steps=state.max_steps,
            comms_active=state.comms_active,
            sensor_noise_std=state.sensor_noise_std,
            prev_distance=state.distance,
            mission_aborted=mission_aborted,
            docked=docked,
        )

        done = new_state.done

        info = {
            "distance_m": new_dist,
            "speed_ms": new_speed,
            "fuel_remaining_kg": fuel_remaining,
            "fuel_consumed_kg": dm,
            "delta_v_ms": dv,
            "docked": docked,
            "mission_aborted": mission_aborted,
            "collision": collision,
            "out_of_bounds": out_of_bounds,
            "los_violation": new_state.los_violation,
        }

        return new_state, done, info

    def compute_hohmann_dv(self, distance: float) -> float:
        """
        Rough estimate of delta-v needed for approach from given distance.
        Used for fuel budget scaling in scenario generator.
        """
        # Simple proportional approximation — not exact Hohmann but useful for scaling
        return 0.1 * distance * self.n
