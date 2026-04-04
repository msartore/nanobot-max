"""Tests for cron service reliability improvements:
- Timer chain protection (try/finally with _arm_timer)
- Missed run detection
- Schedule drift prevention
- Retry with exponential backoff
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nanobot.cron.service import CronService, _compute_next_run, _now_ms
from nanobot.cron.types import CronJob, CronJobState, CronPayload, CronSchedule, CronStore


def _make_cron_service(tmp_path: Path, on_job=None) -> CronService:
    store_path = tmp_path / "cron" / "jobs.json"
    return CronService(store_path=store_path, on_job=on_job)


# ---------------------------------------------------------------------------
# Timer chain protection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_timer_chain_survives_save_failure(tmp_path):
    """When _save_store fails, _arm_timer should still be called in finally block."""
    service = _make_cron_service(tmp_path)
    service.add_job("test", CronSchedule(kind="every", every_ms=50), "message")

    tick_count = {"n": 0}
    save_fail = {"active": False}

    async def on_job(job):
        tick_count["n"] += 1

    service.on_job = on_job

    original_save = service._save_store

    def failing_save():
        if save_fail["active"]:
            raise OSError("disk full")
        return original_save()

    service._save_store = failing_save

    await service.start()

    # Trigger save failures
    save_fail["active"] = True
    await asyncio.sleep(0.2)

    # Jobs should have executed at least once despite save failures
    assert tick_count["n"] >= 1

    save_fail["active"] = False
    service.stop()
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_timer_chain_survives_load_failure(tmp_path):
    """When the store file is corrupted, the service should recover gracefully."""
    service = _make_cron_service(tmp_path)
    service.add_job("test", CronSchedule(kind="every", every_ms=50), "message")
    await service.start()

    # Corrupt the store file after start
    service.store_path.write_text("not valid json", encoding="utf-8")
    service._last_mtime = 0.0

    # Wait for a timer tick which will try to reload
    await asyncio.sleep(0.2)

    # The store should have been reset to empty (not crashed)
    assert service._store is not None

    service.stop()
    await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# Missed run detection
# ---------------------------------------------------------------------------

def test_missed_run_detection_on_startup(tmp_path):
    """When a job missed runs during downtime, it should log a warning."""
    service = _make_cron_service(tmp_path)
    now = _now_ms()

    job = CronJob(
        id="test1",
        name="frequent-job",
        enabled=True,
        schedule=CronSchedule(kind="every", every_ms=60000),
        payload=CronPayload(message="test"),
        state=CronJobState(
            last_run_at_ms=now - 5 * 60000,
        ),
    )
    service._store = CronStore(jobs=[job])
    service._last_mtime = time.time()

    with patch("nanobot.cron.service.logger") as mock_logger:
        service._recompute_next_runs()

    warning_calls = [c for c in mock_logger.warning.call_args_list if "missed" in str(c)]
    assert len(warning_calls) >= 1
    assert "4" in str(warning_calls[0])


def test_no_missed_run_warning_for_first_run(tmp_path):
    """When a job has never run before, no missed run warning."""
    service = _make_cron_service(tmp_path)

    job = CronJob(
        id="test1",
        name="new-job",
        enabled=True,
        schedule=CronSchedule(kind="every", every_ms=60000),
        payload=CronPayload(message="test"),
        state=CronJobState(),
    )
    service._store = CronStore(jobs=[job])
    service._last_mtime = time.time()

    with patch("nanobot.cron.service.logger") as mock_logger:
        service._recompute_next_runs()

    warning_calls = [c for c in mock_logger.warning.call_args_list if "missed" in str(c)]
    assert len(warning_calls) == 0


def test_no_missed_run_warning_for_cron_schedule(tmp_path):
    """Cron schedules don't trigger missed run warnings."""
    service = _make_cron_service(tmp_path)
    now = _now_ms()

    job = CronJob(
        id="test1",
        name="cron-job",
        enabled=True,
        schedule=CronSchedule(kind="cron", expr="*/5 * * * *", tz="UTC"),
        payload=CronPayload(message="test"),
        state=CronJobState(
            last_run_at_ms=now - 3600000,
        ),
    )
    service._store = CronStore(jobs=[job])
    service._last_mtime = time.time()

    with patch("nanobot.cron.service.logger") as mock_logger:
        service._recompute_next_runs()

    warning_calls = [c for c in mock_logger.warning.call_args_list if "missed" in str(c)]
    assert len(warning_calls) == 0


# ---------------------------------------------------------------------------
# Schedule drift prevention
# ---------------------------------------------------------------------------

def test_compute_next_run_from_scheduled_time_not_now():
    """For 'every' schedules, next run is based on the input time."""
    now = _now_ms()
    interval = 60000

    schedule = CronSchedule(kind="every", every_ms=interval)

    past_time = now - 30000
    next_run = _compute_next_run(schedule, past_time)

    expected = past_time + interval
    assert next_run == expected


# ---------------------------------------------------------------------------
# Retry with exponential backoff
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_backoff_on_job_failure(tmp_path):
    """When a job fails, it should retry with exponential backoff."""
    service = _make_cron_service(tmp_path)

    async def on_job(job):
        raise RuntimeError("transient error")

    service.on_job = on_job

    now = _now_ms()
    job = CronJob(
        id="test1",
        name="flaky-job",
        enabled=True,
        schedule=CronSchedule(kind="every", every_ms=60000),
        payload=CronPayload(message="test"),
        state=CronJobState(
            next_run_at_ms=now,
            retry_count=0,
        ),
    )
    service._store = CronStore(jobs=[job])
    service._last_mtime = time.time()

    await service._execute_job(job)

    assert job.state.last_status == "error"
    assert job.state.retry_count == 1
    # Formula: 2^retry_count * 30s = 2^1 * 30 = 60s
    expected_backoff_ms = now + 60 * 1000
    assert job.state.next_run_at_ms >= expected_backoff_ms - 1000


@pytest.mark.asyncio
async def test_retry_count_resets_on_success(tmp_path):
    """When a job succeeds, retry count should reset to 0."""
    service = _make_cron_service(tmp_path)

    async def on_job(job):
        return "ok"

    service.on_job = on_job

    now = _now_ms()
    job = CronJob(
        id="test1",
        name="recovering-job",
        enabled=True,
        schedule=CronSchedule(kind="every", every_ms=60000),
        payload=CronPayload(message="test"),
        state=CronJobState(
            next_run_at_ms=now,
            retry_count=2,
        ),
    )
    service._store = CronStore(jobs=[job])
    service._last_mtime = time.time()

    await service._execute_job(job)

    assert job.state.retry_count == 0
    assert job.state.last_status == "ok"


@pytest.mark.asyncio
async def test_retry_backoff_caps_at_one_hour(tmp_path):
    """Retry backoff should not exceed 1 hour."""
    service = _make_cron_service(tmp_path)

    async def on_job(job):
        raise RuntimeError("persistent error")

    service.on_job = on_job

    now = _now_ms()
    job = CronJob(
        id="test1",
        name="stubborn-job",
        enabled=True,
        schedule=CronSchedule(kind="every", every_ms=60000),
        payload=CronPayload(message="test"),
        state=CronJobState(
            next_run_at_ms=now,
            retry_count=10,
        ),
    )
    service._store = CronStore(jobs=[job])
    service._last_mtime = time.time()

    await service._execute_job(job)

    max_backoff_ms = now + 3600 * 1000
    assert job.state.next_run_at_ms <= max_backoff_ms + 1000


# ---------------------------------------------------------------------------
# Persistence of retry_count
# ---------------------------------------------------------------------------

def test_retry_count_persisted_to_disk(tmp_path):
    """retry_count should be saved and loaded from disk."""
    service = _make_cron_service(tmp_path)

    job = CronJob(
        id="test1",
        name="persisted-job",
        enabled=True,
        schedule=CronSchedule(kind="every", every_ms=60000),
        payload=CronPayload(message="test"),
        state=CronJobState(
            retry_count=5,
        ),
    )
    service._store = CronStore(jobs=[job])
    service._last_mtime = time.time()

    service._save_store()

    service._store = None
    service._load_store()

    loaded_job = service._store.jobs[0]
    assert loaded_job.state.retry_count == 5


# ---------------------------------------------------------------------------
# Arm timer guards against already-done tasks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_arm_timer_skips_done_tasks(tmp_path):
    """_arm_timer should not crash when the old timer task is already done."""
    service = _make_cron_service(tmp_path)

    async def done_task():
        pass

    task = asyncio.create_task(done_task())
    await asyncio.sleep(0)
    assert task.done()

    service._timer_task = task
    service._running = True
    service._store = CronStore(jobs=[])
    service._last_mtime = time.time()

    service._arm_timer()
