"""The Linux install, run and uninstall scripts must at least parse, and the
two decisions that can hurt someone must hold (issue #559).

The scripts themselves are exercised end to end in a clean container (they
download gigabytes and need a GPU, so they cannot run in CI). What a unit test
can pin without side effects: every script is syntactically valid bash, the
entry points are executable, ``uninstall.sh`` refuses a manifest entry that
would aim ``rm -rf`` outside what install creates, and the llama-server
capability probe reports the observed reason, stops the server it started
(also on INT/TERM) and leaves no temp log behind, using a fake server script.
"""

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS = REPO_ROOT / "tools"
ENTRY_POINTS = ("install.sh", "run.sh", "uninstall.sh")
SCRIPTS = (*ENTRY_POINTS, "_common.sh")

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_is_valid_bash(script):
    result = subprocess.run(
        ["bash", "-n", str(TOOLS / script)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_entry_point_is_executable(script):
    assert os.access(TOOLS / script, os.X_OK)


def _env(tmp_path):
    env = dict(os.environ)
    env["MONKEYGRAB_STATE_DIR"] = str(tmp_path / "state")
    env["MONKEYGRAB_MODELS_DIR"] = str(tmp_path / "models")
    return env


def _dry_run(tmp_path, entries):
    """Run ``uninstall.sh --dry-run`` against a manifest of ``entries``."""
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    lines = [f"repo\t{REPO_ROOT}"] + [f"path\t{entry}" for entry in entries]
    (state / "install-manifest.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return subprocess.run(
        [str(TOOLS / "uninstall.sh"), "--dry-run"],
        capture_output=True, text=True, env=_env(tmp_path),
    )


@pytest.mark.parametrize(
    "entry",
    [
        "/",
        "//",
        f"{Path.home()}",
        f"{Path.home()}/",
        f"{Path.home()}/.ssh",
        "/usr/lib",
        "/a/../..",
        f"{REPO_ROOT}/../outside",
        f"{REPO_ROOT}/..",
        "relative/path",
        "./rag/vector_db",
    ],
)
def test_uninstall_refuses_paths_outside_what_install_creates(tmp_path, entry):
    result = _dry_run(tmp_path, [entry])
    assert result.returncode == 0, result.stderr
    assert f"REFUSED (unsafe path in manifest): {entry}\n" in result.stdout


def test_uninstall_accepts_entries_it_may_own_and_normalises_them(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "a.gguf").write_text("x", encoding="utf-8")
    result = _dry_run(
        tmp_path,
        [
            f"{models}/a.gguf",
            f"{models}//sub/../a.gguf/",
            f"{REPO_ROOT}/.venv/",
            f"{REPO_ROOT}/rag/vector_db",
            str(tmp_path / "state" / "llama-server.log"),
        ],
    )
    assert result.returncode == 0, result.stderr
    assert "REFUSED" not in result.stdout, result.stdout
    # trailing slashes and ".." are normalised in the plan
    assert f"  {models}/a.gguf\n" in result.stdout
    assert f"  {REPO_ROOT}/.venv (already gone)\n" in result.stdout or f"  {REPO_ROOT}/.venv\n" in result.stdout
    assert (models / "a.gguf").exists(), "--dry-run must never delete"


def test_uninstall_refuses_a_symlink_that_leaves_the_checkout(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "models" / "link"
    link.parent.mkdir()
    link.symlink_to(outside)
    result = _dry_run(tmp_path, [str(link)])
    assert f"REFUSED (unsafe path in manifest): {link}\n" in result.stdout
    assert outside.exists()


def test_uninstall_refuses_another_checkouts_manifest(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "install-manifest.tsv").write_text("repo\t/somewhere/else\n", encoding="utf-8")
    result = subprocess.run(
        [str(TOOLS / "uninstall.sh"), "--yes"],
        capture_output=True, text=True, env=_env(tmp_path),
    )
    assert result.returncode != 0
    assert "belongs to /somewhere/else" in result.stderr


@pytest.mark.parametrize(
    "banner, expected",
    [
        ("version: 0.3.0-dev (build 10751, commit 3466812d1)", "10751"),
        ("version: 11669 (abc1234)", "11669"),
        ("not a version line", ""),
    ],
)
def test_llama_build_reads_both_version_spellings(tmp_path, banner, expected):
    fake = tmp_path / "llama-server"
    fake.write_text(f"#!/usr/bin/env bash\necho '{banner}'\n", encoding="utf-8")
    fake.chmod(0o755)
    env = _env(tmp_path)
    env["LLAMA_SERVER_BIN"] = str(fake)
    result = subprocess.run(
        ["bash", "-c", f". '{TOOLS / '_common.sh'}'; llama_build"],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


@pytest.mark.parametrize(
    "url, host, port",
    [
        ("http://127.0.0.1:8091", "127.0.0.1", "8091"),
        ("http://localhost", "localhost", "80"),
        ("https://example.org/v1", "example.org", "443"),
        ("http://[::1]:9000", "[::1]", "9000"),
    ],
)
def test_url_host_and_port_default_by_scheme(tmp_path, url, host, port):
    result = subprocess.run(
        ["bash", "-c", f". '{TOOLS / '_common.sh'}'; url_host '{url}'; url_port '{url}'"],
        capture_output=True, text=True, env=_env(tmp_path),
    )
    assert result.stdout.split() == [host, port], result.stderr


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_help_prints_a_usage_block(script):
    result = subprocess.run(
        [str(TOOLS / script), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert result.stdout.startswith("Usage: tools/"), result.stdout


def test_claims_under_symlinked_dirs_can_be_uninstalled(tmp_path):
    """With ~/.local (or any parent) symlinked, what install claims must still
    pass uninstall's resolved allow-list, or uninstall can never finish."""
    real_models, real_state = tmp_path / "real_models", tmp_path / "real_state"
    real_models.mkdir()
    real_state.mkdir()
    (tmp_path / "link_models").symlink_to(real_models)
    (tmp_path / "link_state").symlink_to(real_state)
    env = _env(tmp_path)
    env["MONKEYGRAB_MODELS_DIR"] = str(tmp_path / "link_models")
    env["MONKEYGRAB_STATE_DIR"] = str(tmp_path / "link_state")

    claim = subprocess.run(
        ["bash", "-c", f"""set -euo pipefail
. '{TOOLS / '_common.sh'}'
eval "$(sed -n '/^claim_path()/,/^}}/p' '{TOOLS / 'install.sh'}')"
manifest_add repo "$REPO_ROOT"
claim_path "$MODELS_DIR/fake.gguf"
claim_path "$STATE_DIR/llama-server.log"
"""],
        capture_output=True, text=True, env=env,
    )
    assert claim.returncode == 0, claim.stderr
    (real_models / "fake.gguf").write_text("x", encoding="utf-8")
    (real_state / "llama-server.log").write_text("x", encoding="utf-8")
    manifest = (real_state / "install-manifest.tsv").read_text(encoding="utf-8")
    assert f"path\t{real_models}/fake.gguf\n" in manifest, manifest

    plan = subprocess.run(
        [str(TOOLS / "uninstall.sh"), "--dry-run"], capture_output=True, text=True, env=env
    )
    assert plan.returncode == 0, plan.stderr
    assert "REFUSED" not in plan.stdout, plan.stdout
    assert (real_models / "fake.gguf").exists()

    done = subprocess.run(
        [str(TOOLS / "uninstall.sh"), "--yes"], capture_output=True, text=True, env=env
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert not (real_models / "fake.gguf").exists()
    assert not (real_state / "llama-server.log").exists()
    assert not (real_state / "install-manifest.tsv").exists()


_FAKE_SERVER = """#!/usr/bin/env python3
import http.server, json, os, sys
mode = os.environ["FAKE_MODE"]
if mode == "crash":
    sys.exit(3)
if mode == "unknown-arch":
    print("llama_model_load: error: unknown model architecture: 'gemma-embedding2'", file=sys.stderr)
    sys.exit(1)
args = sys.argv
port = int(args[args.index("--port") + 1])
assert "--mmproj" in args and "--embedding" in args and "-ub" in args
open(os.environ["FAKE_PIDFILE"], "w").write(str(os.getpid()))
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if mode == "never-healthy":
            self.send_response(503); self.end_headers(); return
        body = b"{}" if self.path == "/health" else json.dumps(
            {"modalities": {"vision": mode == "ok"}}).encode()
        self.send_response(200); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
"""


def _probe(tmp_path, mode):
    fake = tmp_path / "llama-server"
    fake.write_text(_FAKE_SERVER, encoding="utf-8")
    fake.chmod(0o755)
    env = _env(tmp_path)
    env.update(
        LLAMA_SERVER_BIN=str(fake), FAKE_MODE=mode, LLAMA_PROBE_TIMEOUT="3",
        FAKE_PIDFILE=str(tmp_path / "pid"),
    )
    return subprocess.run(
        ["bash", "-c", f". '{TOOLS / '_common.sh'}'; llama_probe m.gguf p.gguf"],
        capture_output=True, text=True, env=env, timeout=60,
    )


def test_probe_passes_when_the_server_loads_with_vision_and_is_stopped_after(tmp_path):
    result = _probe(tmp_path, "ok")
    assert result.returncode == 0, result.stderr
    pid = int((tmp_path / "pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize(
    "mode, reason, advises_upgrade",
    [
        ("no-vision", "reports no vision support", False),
        ("crash", "exited during startup", False),
        ("unknown-arch", "exited during startup", True),
        ("never-healthy", "did not answer within 3s", False),
    ],
)
def test_probe_reports_the_observed_reason_and_upgrades_only_when_the_log_says_so(
    tmp_path, mode, reason, advises_upgrade
):
    result = _probe(tmp_path, mode)
    assert result.returncode == 1
    assert reason in result.stderr
    assert ("use a newer build" in result.stderr) is advises_upgrade
    if mode == "unknown-arch":
        assert "unknown model architecture" in result.stderr, "log tail must be shown"
    if mode in ("no-vision", "never-healthy"):
        with pytest.raises(ProcessLookupError):
            os.kill(int((tmp_path / "pid").read_text()), 0)


@pytest.mark.parametrize("value", ["abc", "0", "08", "-5", "1.5"])
def test_probe_rejects_a_bad_timeout_before_starting_anything(tmp_path, value):
    fake = tmp_path / "llama-server"
    fake.write_text("#!/usr/bin/env bash\ntouch started\n", encoding="utf-8")
    fake.chmod(0o755)
    env = _env(tmp_path)
    env.update(LLAMA_SERVER_BIN=str(fake), LLAMA_PROBE_TIMEOUT=value)
    result = subprocess.run(
        ["bash", "-c", f"cd '{tmp_path}'; . '{TOOLS / '_common.sh'}'; llama_probe m p"],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 1
    assert "LLAMA_PROBE_TIMEOUT" in result.stderr
    assert not (tmp_path / "started").exists()


def test_probe_restores_the_callers_traps(tmp_path):
    fake = tmp_path / "llama-server"
    fake.write_text(_FAKE_SERVER, encoding="utf-8")
    fake.chmod(0o755)
    env = _env(tmp_path)
    env.update(LLAMA_SERVER_BIN=str(fake), FAKE_MODE="ok", FAKE_PIDFILE=str(tmp_path / "pid"))
    result = subprocess.run(
        ["bash", "-c", f"trap 'echo caller-exit-trap' EXIT; . '{TOOLS / '_common.sh'}'; llama_probe m p; trap -p EXIT"],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "trap -- 'echo caller-exit-trap' EXIT" in result.stdout
    assert result.stdout.rstrip().endswith("caller-exit-trap")


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_a_signal_during_the_probe_leaves_no_orphan_and_no_temp_log(tmp_path, sig):
    fake = tmp_path / "llama-server"
    fake.write_text(_FAKE_SERVER, encoding="utf-8")
    fake.chmod(0o755)
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    marker = tmp_path / "caller-exit-trap-ran"
    env = _env(tmp_path)
    env.update(
        LLAMA_SERVER_BIN=str(fake), FAKE_MODE="never-healthy", LLAMA_PROBE_TIMEOUT="60",
        FAKE_PIDFILE=str(tmp_path / "pid"), TMPDIR=str(scratch),
    )
    script = subprocess.Popen(
        ["bash", "-c", f"trap 'touch \"{marker}\"' EXIT; . '{TOOLS / '_common.sh'}'; llama_probe m p"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            if (tmp_path / "pid").exists() and (tmp_path / "pid").read_text():
                break
            time.sleep(0.1)
        server_pid = int((tmp_path / "pid").read_text())
        os.kill(server_pid, 0)  # alive mid-probe
        script.send_signal(sig)
        assert script.wait(timeout=30) in (128 + sig, -sig)
    finally:
        if script.poll() is None:
            script.kill()
    with pytest.raises(ProcessLookupError):
        os.kill(server_pid, 0)
    assert list(scratch.iterdir()) == [], "the temp log must be removed"
    assert marker.exists(), "the caller's own EXIT trap must still run after the signal"


def test_check_reports_the_probe_pending_until_the_ggufs_exist(tmp_path):
    result = subprocess.run(
        [str(TOOLS / "install.sh"), "--check"],
        capture_output=True, text=True, env=_env(tmp_path),
    )
    assert "probe llama-server with the downloaded model (pending" in result.stdout
    assert "too old" not in result.stdout
