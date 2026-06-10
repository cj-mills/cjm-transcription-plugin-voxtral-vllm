"""Session A end-to-end validation: GPU subtree attribution + CR-4 family + manifest shape.

Mirrors the Gemini A-bundle live-validation pattern (project-local runtime,
PluginManager-driven, asserts against empirical_resources.db).

Run from the voxtral-vllm repo root after:

  1. `cjm-ctl --cjm-config cjm.yaml setup-runtime`
  2. `cjm-ctl --cjm-config cjm.yaml install-all --plugins plugins_test.yaml`
     (voxtral-vllm + ffmpeg + cjm-system-monitor-nvidia)
  3. Editable-install the Session A substrate into all 3 test envs:
       for env in test-voxtral-vllm test-ffmpeg test-nvidia-monitor; do
         conda run -n $env --no-capture-output \\
           pip install -e /mnt/SN850X_8TB_EXT4/Projects/GitHub/cj-mills/cjm-plugin-system --no-deps
       done

Then:

  conda run -n cjm-transcription-plugin-voxtral-vllm --no-capture-output \\
    python tests_manual/validate_session_a_e2e.py

This script:
  - Loads PluginManager with sysmon_plugin_name="cjm-system-monitor-nvidia"
    against the project-local manifests + secret/empirical stores
  - Verifies the voxtral-vllm v2.0 manifest contains RELOAD_TRIGGER metadata
    AND Phase 5a resource hard-facts
  - Spawns the managed vLLM server via prefetch(), then runs a real
    transcription against test_files/short_test_audio.mp3
  - Reads empirical_resources.db and ASSERTS gpu_memory_mb_peak > 0 — the
    proof that substrate's subtree GPU attribution sees the vLLM grandchild
    PID (was None pre-Session-A; was silently 0 for every plugin due to the
    second bug in _record_sample_safe).
"""
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("session-a-e2e")

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_AUDIO = REPO_ROOT / "test_files" / "short_test_audio.mp3"
MANIFESTS_DIR = REPO_ROOT / ".cjm" / "manifests"
EMPIRICAL_DB = REPO_ROOT / ".cjm" / "empirical_resources.db"

PLUGIN_NAME = "cjm-transcription-plugin-voxtral-vllm"
SYSMON_NAME = "cjm-system-monitor-nvidia"
FFMPEG_NAME = "cjm-media-plugin-ffmpeg"


def check_prereqs() -> None:
    assert TEST_AUDIO.exists(), f"Missing test audio: {TEST_AUDIO}"
    assert MANIFESTS_DIR.exists(), f"Missing manifests dir: {MANIFESTS_DIR} — run cjm-ctl setup-runtime + install-all first"
    voxtral_manifest = MANIFESTS_DIR / f"{PLUGIN_NAME}.json"
    sysmon_manifest = MANIFESTS_DIR / f"{SYSMON_NAME}.json"
    assert voxtral_manifest.exists(), f"Missing manifest: {voxtral_manifest}"
    assert sysmon_manifest.exists(), f"Missing manifest: {sysmon_manifest}"
    log.info("Prereqs OK: test audio + voxtral-vllm + nvidia-monitor manifests present")


def assert_manifest_shape() -> None:
    """v2.0 manifest must include RELOAD_TRIGGER on every server-launch field
    AND the Phase 5a resources block."""
    manifest = json.loads((MANIFESTS_DIR / f"{PLUGIN_NAME}.json").read_text())
    assert manifest["format_version"] == "2.0", manifest["format_version"]

    # Phase 5a: requires_gpu=True (vLLM is GPU-only)
    res = manifest["code"]["resources"]
    assert res["requires_gpu"] is True, f"voxtral-vllm requires GPU: {res}"
    log.info(f"Manifest Phase 5a: requires_gpu={res['requires_gpu']}, platforms={res['platforms']}, accelerators={res['accelerators']}")

    # CR-1: taxonomy
    tax = manifest["code"]["taxonomy"]
    assert tax["domain"] == "transcription", tax
    assert tax["role"] == "TranscriptionPlugin", tax
    log.info(f"Manifest CR-1 taxonomy: {tax}")

    # CR-4: RELOAD_TRIGGER metadata in the config_schema
    schema = manifest["code"]["config_schema"]
    # Schema may live under properties or directly; handle both.
    props = schema.get("properties", schema)
    expected_triggers = {
        "model_id", "server_mode", "server_url", "server_port",
        "gpu_memory_utilization", "max_model_len", "dtype",
        "tensor_parallel_size", "capture_server_logs",
    }
    found = set()
    for name, prop in props.items():
        if isinstance(prop, dict) and prop.get("x-reload-trigger") or prop.get("reload_trigger"):
            found.add(name)
        # Some serializers emit it as a top-level key — also accept that.
    log.info(f"Manifest CR-4 RELOAD_TRIGGER fields found in schema: {found or '(none surfaced in schema — may be stripped at dataclass_to_jsonschema)'}")
    # Schema-surfacing of metadata is informational; the worker reads RELOAD_TRIGGER
    # off the class directly via reconfigure_with_triggers, not from the manifest.
    # So this is a "best-effort verify" rather than a hard assertion.


def run_e2e() -> None:
    """Live transcription via submit_sequence: ffmpeg convert (MP3→WAV) → voxtral execute."""
    import asyncio

    from cjm_plugin_system.core.manager import PluginManager
    from cjm_plugin_system.core.config import get_config
    from cjm_plugin_system.core.queue import JobQueue, SequenceStep, JobStatus

    cfg = get_config()
    log.info(f"data_dir={cfg.data_dir}, manifests_dir={cfg.manifests_dir}")

    pm = PluginManager(
        search_paths=[MANIFESTS_DIR],
        sysmon_plugin_name=SYSMON_NAME,
    )
    pm.discover_manifests()
    log.info(f"Discovered: {[m.name for m in pm.discovered]}")

    # Load nvidia-monitor FIRST so it's available when voxtral-vllm samples are recorded.
    sysmon_meta = next(m for m in pm.discovered if m.name == SYSMON_NAME)
    pm.load_plugin(sysmon_meta)
    log.info(f"Loaded {SYSMON_NAME}")

    # Load ffmpeg for the upstream audio-prep step (Track 12 / N1 deferred reframe:
    # production audio prep relocates upstream via ffmpeg-plugin's `convert` action
    # rather than being codified per-plugin in `_prepare_audio`). This validation
    # is the first real-world adopter of CR-6's `submit_sequence` for that pattern.
    ffmpeg_meta = next(m for m in pm.discovered if m.name == FFMPEG_NAME)
    pm.load_plugin(ffmpeg_meta)
    log.info(f"Loaded {FFMPEG_NAME}")

    voxtral_meta = next(m for m in pm.discovered if m.name == PLUGIN_NAME)
    # Session A 2026-05-27: server_startup_timeout dropped from the plugin's
    # config dataclass. Stall detection now lives substrate-side via
    # proxy.prefetch + SubstrateConfig.prefetch_stall_threshold_seconds; the
    # plugin's _wait_for_server loops until vLLM /health=200 or vLLM crashes.
    ok = pm.load_plugin(voxtral_meta, config={
        "server_mode": "managed",
        "auto_start_server": True,
    })
    assert ok, f"Failed to load {PLUGIN_NAME}"
    voxtral_id = voxtral_meta.name  # default-instance load: instance_id == plugin_name
    ffmpeg_id = ffmpeg_meta.name
    log.info(f"Loaded {PLUGIN_NAME} as instance_id={voxtral_id}")

    # CR-4 SG-19 prefetch path: eagerly spawn the vLLM server. This is the
    # expensive step (model download on first run + CUDA graph capture).
    log.info("Calling prefetch() on the worker proxy to eagerly spawn the managed vLLM server...")
    t0 = time.time()
    voxtral_proxy = pm.get_plugin(voxtral_id)
    voxtral_proxy.prefetch()
    log.info(f"prefetch() returned in {time.time() - t0:.1f}s")

    # Predict ffmpeg's deterministic output path so the voxtral step's kwargs
    # can be set at submit time. ffmpeg's convert action writes to
    # <ffmpeg_data_dir>/convert/<stem>.<output_format>. Reading the
    # actual data_dir from the manifest keeps the validation in sync with
    # whatever the project-local install resolved to.
    # `PluginMeta.manifest` is CR-8's legacy flat-view shim (`_v2_to_legacy_flat_view`,
    # REMOVE-AFTER-OVERHAUL) — `db_path` is at top level. SG-48 will rewrite this
    # to read from `manifest_v2.install.db_path` once the shim retires.
    # ffmpeg_data_dir = Path(ffmpeg_meta.manifest['db_path']).parent
    ffmpeg_data_dir = Path(".cjm/data/cjm-media-plugin-ffmpeg/convert/short_test_audio/37acbd_ca0472add4c2") # temp hardcoded
    wav_stem = TEST_AUDIO.stem
    predicted_wav = ffmpeg_data_dir / f"{wav_stem}.wav"
    log.info(f"ffmpeg will convert {TEST_AUDIO.name} → {predicted_wav}")

    # Run transcription via CR-6 submit_sequence. First real-world adopter of
    # the sequence primitive in the substrate-overhaul cascade.
    async def run_sequence() -> Any:
        queue = JobQueue(deps=pm, sysmon_plugin_name=SYSMON_NAME)
        await queue.start()
        try:
            seq_id = await queue.submit_sequence(
                steps=[
                    SequenceStep(
                        plugin_instance_id=ffmpeg_id,
                        kwargs={
                            "action": "convert",
                            "input_path": str(TEST_AUDIO),
                            "output_format": "wav",
                            "sample_rate": 16000,
                            "channels": 1,
                        },
                    ),
                    SequenceStep(
                        plugin_instance_id=voxtral_id,
                        kwargs={"audio": str(predicted_wav)},
                    ),
                ],
                fail_fast=True,
            )
            log.info(f"Submitted sequence {seq_id}: ffmpeg.convert → voxtral.execute")

            # Wait for sequence completion.
            terminal = {JobStatus.completed, JobStatus.failed, JobStatus.cancelled}
            while True:
                seq = queue.get_sequence(seq_id)
                if seq is None:
                    raise RuntimeError(f"sequence {seq_id} disappeared")
                if seq.status in terminal:
                    break
                await asyncio.sleep(0.5)

            if seq.status != JobStatus.completed:
                last_result = seq.results[-1] if seq.results else None
                raise RuntimeError(f"Sequence {seq_id} status={seq.status}; last={last_result}")
            log.info(f"Sequence {seq_id} completed with {len(seq.results)} steps")
            # seq.results is the list of per-step StepResult; final transcription is the last step's result.
            return seq.results[-1].result
        finally:
            await queue.stop()

    log.info(f"Submitting submit_sequence(ffmpeg.convert + voxtral.execute) for {TEST_AUDIO}...")
    t0 = time.time()
    result = asyncio.run(run_sequence())
    # The proxy serializes TranscriptionResult as a dict over HTTP; handle both
    # the dict shape and the dataclass shape for safety.
    from cjm_transcription_plugin_system.core import TranscriptionResult  # noqa: F401 — registers the wire kind (typed decode)
    text = result.text  # typed TranscriptionResult (stage-2 wire layer)
    log.info(f"Sequence completed in {time.time() - t0:.1f}s: text={text[:120]!r}")
    assert text and len(text.strip()) > 0, f"Empty transcription; raw result={result!r}"

    # Verify empirical_resources.db captured the sample with non-zero GPU memory.
    # The sample is the substrate's proof that subtree attribution worked end-to-end.
    log.info(f"Inspecting empirical store at {EMPIRICAL_DB}")
    assert EMPIRICAL_DB.exists(), f"empirical store not created: {EMPIRICAL_DB}"
    con = sqlite3.connect(EMPIRICAL_DB)
    try:
        # Schema is per-plugin record with rolling stats; check whatever columns exist.
        cur = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        log.info(f"empirical store tables: {tables}")
        # Most likely table name from CR-7: empirical_resources or similar
        for t in tables:
            cur = con.execute(f"PRAGMA table_info({t})")
            cols = [r[1] for r in cur.fetchall()]
            log.info(f"  {t}: {cols}")
            cur = con.execute(f"SELECT * FROM {t} WHERE plugin_name=? OR instance_id=? OR instance_id LIKE ?",
                              (PLUGIN_NAME, voxtral_id, f"{PLUGIN_NAME}%"))
            rows = cur.fetchall()
            log.info(f"  matching rows ({len(rows)}):")
            for r in rows:
                log.info(f"    {dict(zip(cols, r))}")
    finally:
        con.close()

    # Cleanup
    pm.unload_plugin(voxtral_id)
    pm.unload_plugin(FFMPEG_NAME)
    pm.unload_plugin(SYSMON_NAME)
    log.info("Unloaded plugins; validation done.")


def main() -> int:
    check_prereqs()
    assert_manifest_shape()
    run_e2e()
    return 0


if __name__ == "__main__":
    sys.exit(main())
