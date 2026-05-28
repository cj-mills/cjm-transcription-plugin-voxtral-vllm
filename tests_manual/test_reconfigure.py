"""CR-4 reconfigure-lifecycle validation for the Voxtral vLLM plugin.

Contract-level (no real vLLM server spawn — server startup is GPU-heavy + slow;
the wiring is what we validate). Exercises the substrate's reconfigure delta
path in-process with a fake VLLMServer object:

  1. reconfigure(model_id change) -> RELEASE the server (RELOAD_TRIGGER ->
     _release_vllm_server) + RE-APPLY config (_apply_config) -- the behavior
     CR-4 wired into PluginManager.update_plugin_config
  2. a non-trigger field change (temperature) must NOT release the server
  3. a server-launch field (gpu_memory_utilization) is also a RELOAD_TRIGGER
     (was a silent no-op pre-CR-4 because the old initialize() only diff-checked
     model_id/server_mode/server_port)
  4. on_disable releases the server (CR-2)
  5. server_mode managed -> external also releases (and the OpenAI client gets
     rebuilt with the external base_url)

Requires the substrate version with the two-phase reconfigure (CR-4) AND the
Session A telemetry commit (for the worker-env wiring). Run from the repo root
in the plugin's env:

    conda run -n cjm-transcription-plugin-voxtral-vllm --no-capture-output \\
        python tests_manual/test_reconfigure.py

Becomes a pytest under Track 17. A real-server + real-audio variant belongs in
a GPU-marked pytest (test_files/).
"""
import sys

# External-mode base config: lets _apply_config skip VLLMServer construction
# so we don't need vllm installed or a GPU for the lifecycle assertions below.
EXTERNAL = {
    "model_id": "mistralai/Voxtral-Mini-3B-2507",
    "server_mode": "external",
    "server_url": "http://localhost:8000",
}

# Managed-mode base config: triggers VLLMServer wrapper construction without
# actually spawning the subprocess (start() is deferred to prefetch / first execute).
MANAGED = {
    "model_id": "mistralai/Voxtral-Mini-3B-2507",
    "server_mode": "managed",
    "server_port": 8000,
    "gpu_memory_utilization": 0.85,
    "max_model_len": 32768,
    "dtype": "auto",
    "tensor_parallel_size": 1,
    "capture_server_logs": True,
}


class _FakeRunningServer:
    """Stand-in for VLLMServer that reports as running without a Popen.

    The real VLLMServer spawns a `python -m vllm.entrypoints.openai.api_server`
    subprocess at start(); the lifecycle test only needs to verify that
    `_release_vllm_server` invokes `.stop()` and clears `self.server`.
    """
    def __init__(self):
        self.stop_calls = 0
        self._running = True

    def is_running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self.stop_calls += 1
        self._running = False


def main() -> int:
    from cjm_transcription_plugin_voxtral_vllm.plugin import VoxtralVLLMPlugin

    p = VoxtralVLLMPlugin()
    # First-time setup via external mode so no managed-server wrapper is built.
    p._apply_config(EXTERNAL)
    assert p.config.model_id == "mistralai/Voxtral-Mini-3B-2507"
    assert p.config.server_mode == "external"
    assert p.server is None, "external mode must not construct a VLLMServer"
    assert p.client is not None, "OpenAI client must be built in both modes"

    # 1) model_id trigger releases + re-applies (server_mode stays external so
    #    server stays None; the OpenAI client is rebuilt though — verify via
    #    identity change).
    old_client = p.client
    p.reconfigure(EXTERNAL, {**EXTERNAL, "model_id": "mistralai/Voxtral-Small-24B-2507"})
    assert p.config.model_id == "mistralai/Voxtral-Small-24B-2507"
    assert p.client is not None and p.client is not old_client, (
        "RELOAD_TRIGGER must rebuild the client via _release_vllm_server + _apply_config"
    )
    print("[1] reconfigure model_id: released + re-applied  OK")

    # 2) non-trigger field (temperature) retains the client
    same_client = p.client
    same_config = {**EXTERNAL, "model_id": "mistralai/Voxtral-Small-24B-2507"}
    p.reconfigure(same_config, {**same_config, "temperature": 0.5})
    assert p.client is same_client, (
        "non-trigger (temperature) change must NOT rebuild the client"
    )
    assert p.config.temperature == 0.5
    print("[2] temperature change (non-trigger): client retained + applied  OK")

    # 3) Switch to managed mode for the server-launch field tests.
    p._apply_config(MANAGED)
    assert p.config.server_mode == "managed"
    assert p.server is not None, "managed mode must construct a VLLMServer wrapper"
    # Swap in a fake server so we can observe release without spawning a Popen.
    p.server = _FakeRunningServer()

    # gpu_memory_utilization is a server-launch arg; CR-4 fires the trigger.
    # (Pre-CR-4 init() only diff-checked model_id/server_mode/server_port, so
    # a gpu_memory_utilization change was silently a no-op.)
    fake_server = p.server
    p.reconfigure(MANAGED, {**MANAGED, "gpu_memory_utilization": 0.5})
    assert fake_server.stop_calls == 1, (
        "gpu_memory_utilization RELOAD_TRIGGER must release the existing server"
    )
    assert p.server is not None and p.server is not fake_server, (
        "_apply_config must rebuild the VLLMServer wrapper with new args"
    )
    assert p.config.gpu_memory_utilization == 0.5
    print("[3] gpu_memory_utilization change (server-launch trigger): released + re-applied  OK")

    # 4) on_disable releases (CR-2)
    p.server = _FakeRunningServer()
    fake_server = p.server
    p.on_disable()
    assert fake_server.stop_calls == 1, "on_disable must release the server"
    assert p.server is None
    print("[4] on_disable: server released  OK")

    # 5) managed -> external mode flip releases the managed server AND rebuilds
    # the client against the external base_url. _apply_config then skips
    # VLLMServer construction (external mode).
    p._apply_config(MANAGED)
    p.server = _FakeRunningServer()
    fake_server = p.server
    p.reconfigure(MANAGED, EXTERNAL)
    assert fake_server.stop_calls == 1
    assert p.server is None, "external mode must not construct a VLLMServer"
    assert p.client is not None and "localhost:8000/v1" in str(p.client.base_url)
    print("[5] managed -> external mode flip: released + client rebuilt  OK")

    print("RECONFIGURE VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
