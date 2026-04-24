"""
Spacecraft Rendezvous Environment Client
=========================================
Typed async HTTP client for connecting to the OpenEnv server.
Used by the GRPO training notebook to connect to HuggingFace Spaces.

IMPORTANT: Client never imports server internals (OpenEnv requirement).
All communication is via HTTP only.
"""

import asyncio
import httpx
import json
from typing import Optional, Dict, Any

from models import (
    RendezvousAction, RendezvousObservation, RendezvousState,
    ResetRequest, StepRequest, StepResponse,
)


class SpacecraftEnvClient:
    """
    Async client for the Spacecraft Rendezvous OpenEnv environment.

    Usage (in Colab training loop):
        client = SpacecraftEnvClient("https://your-space.hf.space")
        obs = await client.reset(seed=42, difficulty="easy")
        while not obs.done:
            action = RendezvousAction(fx=-0.5, fy=0.8, reasoning="...")
            resp = await client.step(action)
            obs = resp.observation
        grade = await client.grade()
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7860",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Use 'async with SpacecraftEnvClient(...) as client:'")
        return self._client

    async def health(self) -> Dict[str, Any]:
        r = await self._get_client().get("/health")
        r.raise_for_status()
        return r.json()

    async def reset(
        self,
        seed: int = 42,
        difficulty: Optional[str] = None,
    ) -> RendezvousObservation:
        payload = ResetRequest(seed=seed, difficulty=difficulty)
        r = await self._get_client().post(
            "/reset",
            content=payload.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        return RendezvousObservation(**r.json())

    async def step(self, action: RendezvousAction) -> StepResponse:
        payload = StepRequest(action=action)
        r = await self._get_client().post(
            "/step",
            content=payload.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        data = r.json()
        return StepResponse(
            observation=RendezvousObservation(**data["observation"]),
            reward=data["reward"],
            done=data["done"],
            info=data["info"],
        )

    async def state(self) -> RendezvousState:
        r = await self._get_client().get("/state")
        r.raise_for_status()
        return RendezvousState(**r.json())

    async def grade(self) -> Dict[str, Any]:
        r = await self._get_client().post("/grade")
        r.raise_for_status()
        return r.json()

    async def baseline(self) -> Dict[str, Any]:
        r = await self._get_client().get("/baseline")
        r.raise_for_status()
        return r.json()

    async def info(self) -> Dict[str, Any]:
        r = await self._get_client().get("/info")
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Sync wrapper for environments that prefer sync calls (e.g. TRL rollout fn)
# ---------------------------------------------------------------------------

class SpacecraftEnvClientSync:
    """
    Synchronous wrapper around SpacecraftEnvClient.
    Use this in TRL/Unsloth training loops that don't use asyncio.
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._http = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def health(self) -> Dict[str, Any]:
        return self._http.get("/health").json()

    def reset(self, seed: int = 42, difficulty: Optional[str] = None) -> RendezvousObservation:
        payload = ResetRequest(seed=seed, difficulty=difficulty)
        r = self._http.post(
            "/reset",
            content=payload.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        return RendezvousObservation(**r.json())

    def step(self, action: RendezvousAction) -> StepResponse:
        payload = StepRequest(action=action)
        r = self._http.post(
            "/step",
            content=payload.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        data = r.json()
        return StepResponse(
            observation=RendezvousObservation(**data["observation"]),
            reward=data["reward"],
            done=data["done"],
            info=data["info"],
        )

    def grade(self) -> Dict[str, Any]:
        return self._http.post("/grade").json()

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
