"""Phase 10: 16-client concurrent load test against the deployed HF Space.

Target: ``https://arun-sanjay-red-button.hf.space`` (overridable via
``HF_SPACE_API_URL``). GRPO training (PROJECT.md Section 19) issues
``num_generations=4`` x ``gradient_accumulation_steps=4`` = 16 simultaneous
environment rollouts per training batch. Before any real training run we must
verify the Space can serve 16 parallel WebSocket sessions without session
crosstalk, errors, or unacceptable latency degradation.

Success criteria (all four must pass):

1. ``error_count == 0`` — every client completes its episode.
2. ``unique_episode_ids == success_count`` — full session isolation.
3. ``not any_tampering`` — no tampering crosstalk (all clients only
   ``read_file``; a tampering event anywhere means Phase 7's
   ``EpisodeContext`` isolation leaks under load).
4. ``latency_degradation_factor < 5.0`` — mean 16-client step latency is
   less than 5x the warm-up single-client mean.

The script measures a warm-up episode first (``client_id=-1``) to absorb
HF cold-start cost and then ``asyncio.gather``s 16 ``run_episode`` coroutines.
Each coroutine wraps the sync ``ShutdownGymClient`` session in
``asyncio.to_thread`` (the client is sync per Phase 8/9). Per-client and
aggregate results are written to ``results/phase10_concurrency.json``.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from red_button.client import ShutdownGymClient
from red_button.models import ShutdownAction

HF_URL = os.environ.get("HF_SPACE_API_URL", "https://arun-sanjay-red-button.hf.space")
N_CLIENTS = int(os.environ.get("N_CLIENTS", "16"))
TURNS_PER_EPISODE = int(os.environ.get("TURNS_PER_EPISODE", "10"))
TIER = int(os.environ.get("TIER", "2"))

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_JSON = RESULTS_DIR / "phase10_concurrency.json"


@dataclass
class ClientResult:
    client_id: int
    episode_id: Optional[str] = None
    turn_count: int = 0
    tier: int = TIER
    total_duration_s: float = 0.0
    mean_step_latency_ms: float = 0.0
    max_step_latency_ms: float = 0.0
    error: Optional[str] = None
    tampering_events: list[str] = field(default_factory=list)
    # Kept out of the success-criteria aggregation but useful in the JSON so a
    # human reading the artifact can see the shape of the latency distribution.
    step_latencies_ms: list[float] = field(default_factory=list)


def _drive_session(
    client_id: int, base_url: str, tier: int, turns: int
) -> ClientResult:
    """Blocking per-client session — runs inside ``asyncio.to_thread``."""
    result = ClientResult(client_id=client_id, tier=tier)
    session_t0 = time.perf_counter()
    try:
        sync_env = ShutdownGymClient(base_url=base_url).sync()
        with sync_env:
            reset_result = sync_env.reset(tier=tier)
            obs = reset_result.observation
            assert obs is not None, "reset returned None observation"
            assert obs.state is not None, "reset observation missing nested state"
            result.episode_id = obs.state.episode_id

            for _ in range(turns):
                step_t0 = time.perf_counter()
                step_result = sync_env.step(
                    ShutdownAction(
                        tool_name="read_file",
                        arguments={"path": "/sandbox/problems.json"},
                    )
                )
                step_t1 = time.perf_counter()
                result.step_latencies_ms.append((step_t1 - step_t0) * 1000.0)

                # A benign read_file shouldn't flip done; if it does something
                # went wrong — surface it as an error rather than silently
                # truncating the run.
                if step_result.done:
                    raise RuntimeError(
                        f"step unexpectedly done=True after "
                        f"{len(result.step_latencies_ms)} turns"
                    )

            state = sync_env.state()
            result.turn_count = state.turn_count
            result.tampering_events = list(state.tampering_events)
    except Exception as e:  # noqa: BLE001 — we want every failure mode captured
        result.error = f"{type(e).__name__}: {e}"
    finally:
        result.total_duration_s = time.perf_counter() - session_t0
        if result.step_latencies_ms:
            result.mean_step_latency_ms = statistics.mean(result.step_latencies_ms)
            result.max_step_latency_ms = max(result.step_latencies_ms)
    return result


async def run_episode(client_id: int) -> ClientResult:
    """Async wrapper so we can ``asyncio.gather`` 16 blocking sessions."""
    return await asyncio.to_thread(
        _drive_session, client_id, HF_URL, TIER, TURNS_PER_EPISODE
    )


def _fmt_client_line(r: ClientResult) -> str:
    if r.error is not None:
        return (
            f"  [client {r.client_id:>2}] ERROR  duration={r.total_duration_s:6.2f}s  "
            f"{r.error}"
        )
    return (
        f"  [client {r.client_id:>2}] OK     duration={r.total_duration_s:6.2f}s  "
        f"turns={r.turn_count:>2}  mean_step={r.mean_step_latency_ms:7.1f}ms  "
        f"max_step={r.max_step_latency_ms:7.1f}ms  ep={r.episode_id}"
    )


async def main() -> int:
    print(f"[phase10] target={HF_URL}")
    print(
        f"[phase10] N_CLIENTS={N_CLIENTS} TURNS_PER_EPISODE={TURNS_PER_EPISODE} "
        f"TIER={TIER}"
    )

    # 1. Warm-up single client — absorb container cold-start cost so the
    #    16-client burst isn't inflated by HF's first-request penalty.
    print("[phase10] warm-up (single client, client_id=-1)...")
    warmup = await run_episode(-1)
    print(_fmt_client_line(warmup))
    if warmup.error is not None:
        # If the warm-up itself fails we almost certainly can't trust a burst
        # result, so surface it loudly but continue — the failure data is part
        # of the artifact.
        print("[phase10] WARNING: warm-up failed, proceeding with burst anyway")

    # 2. 16-client burst.
    print(f"[phase10] burst: dispatching {N_CLIENTS} concurrent clients...")
    burst_t0 = time.perf_counter()
    results: list[ClientResult] = await asyncio.gather(
        *[run_episode(i) for i in range(N_CLIENTS)]
    )
    wall_clock_duration = time.perf_counter() - burst_t0

    for r in results:
        print(_fmt_client_line(r))

    # 3. Aggregate stats.
    successes = [r for r in results if r.error is None]
    success_count = len(successes)
    error_count = N_CLIENTS - success_count
    unique_episode_ids = len({r.episode_id for r in successes if r.episode_id})
    any_tampering = any(r.tampering_events for r in successes)

    all_step_latencies = [ms for r in successes for ms in r.step_latencies_ms]
    aggregate_mean_step_ms = (
        statistics.mean(all_step_latencies) if all_step_latencies else 0.0
    )
    aggregate_max_step_ms = max(all_step_latencies) if all_step_latencies else 0.0

    warmup_mean = warmup.mean_step_latency_ms or 0.0
    latency_degradation_factor = (
        (aggregate_mean_step_ms / warmup_mean) if warmup_mean > 0 else float("inf")
    )

    total_session_seconds = sum(r.total_duration_s for r in successes)
    actual_concurrency_factor = (
        (total_session_seconds / wall_clock_duration)
        if wall_clock_duration > 0
        else 0.0
    )

    # 4. Success criteria.
    crit_no_errors = error_count == 0
    crit_isolation = unique_episode_ids == success_count
    crit_no_tampering = not any_tampering
    crit_latency = latency_degradation_factor < 5.0

    print("")
    print("[phase10] ==== AGGREGATE ====")
    print(f"  success_count              = {success_count}/{N_CLIENTS}")
    print(f"  error_count                = {error_count}")
    print(f"  unique_episode_ids         = {unique_episode_ids}")
    print(f"  any_tampering              = {any_tampering}")
    print(f"  warmup_mean_step_ms        = {warmup_mean:.1f}")
    print(f"  warmup_max_step_ms         = {warmup.max_step_latency_ms:.1f}")
    print(f"  aggregate_mean_step_ms     = {aggregate_mean_step_ms:.1f}")
    print(f"  aggregate_max_step_ms      = {aggregate_max_step_ms:.1f}")
    print(f"  latency_degradation_factor = {latency_degradation_factor:.2f}x")
    print(f"  wall_clock_duration_s      = {wall_clock_duration:.2f}")
    print(f"  warmup_duration_s          = {warmup.total_duration_s:.2f}")
    print(f"  actual_concurrency_factor  = {actual_concurrency_factor:.2f}x")
    print("")
    print("[phase10] ==== SUCCESS CRITERIA ====")
    print(f"  (1) error_count == 0                     -> {'PASS' if crit_no_errors else 'FAIL'}")
    print(f"  (2) unique_episode_ids == success_count  -> {'PASS' if crit_isolation else 'FAIL'}")
    print(f"  (3) not any_tampering                    -> {'PASS' if crit_no_tampering else 'FAIL'}")
    print(f"  (4) latency_degradation_factor < 5.0     -> {'PASS' if crit_latency else 'FAIL'}")

    all_pass = crit_no_errors and crit_isolation and crit_no_tampering and crit_latency
    print("")
    print(f"[phase10] overall: {'PASS' if all_pass else 'FAIL'}")

    # 5. Save JSON artifact.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "hf_url": HF_URL,
            "n_clients": N_CLIENTS,
            "turns_per_episode": TURNS_PER_EPISODE,
            "tier": TIER,
        },
        "warmup": asdict(warmup),
        "clients": [asdict(r) for r in results],
        "aggregate": {
            "success_count": success_count,
            "error_count": error_count,
            "unique_episode_ids": unique_episode_ids,
            "any_tampering": any_tampering,
            "warmup_mean_step_ms": warmup_mean,
            "warmup_max_step_ms": warmup.max_step_latency_ms,
            "aggregate_mean_step_ms": aggregate_mean_step_ms,
            "aggregate_max_step_ms": aggregate_max_step_ms,
            "latency_degradation_factor": latency_degradation_factor,
            "wall_clock_duration_s": wall_clock_duration,
            "warmup_duration_s": warmup.total_duration_s,
            "actual_concurrency_factor": actual_concurrency_factor,
        },
        "success_criteria": {
            "no_errors": crit_no_errors,
            "session_isolation": crit_isolation,
            "no_tampering": crit_no_tampering,
            "latency_under_5x": crit_latency,
            "overall_pass": all_pass,
        },
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2))
    print(f"[phase10] wrote {RESULTS_JSON}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    rc = asyncio.run(main())
    if rc != 0:
        raise SystemExit(rc)
