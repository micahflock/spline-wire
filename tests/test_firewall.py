"""Windows Firewall helper: the cached watcher everywhere, the real scripts on Windows."""
import json
import subprocess
import sys
import time

import pytest

from splinewire.webapp import firewall
from splinewire.webapp.firewall import Firewall, phones_can_connect


def wait(fw: Firewall, timeout: float = 5.0) -> dict:
    end = time.monotonic() + timeout
    while fw.snapshot()["busy"] and time.monotonic() < end:
        time.sleep(0.01)
    return fw.snapshot()


def test_unsupported_off_windows():
    if sys.platform == "win32":
        pytest.skip("Windows has a firewall to check")
    assert firewall.status() == {"supported": False}
    assert firewall.request_allow() is False
    assert phones_can_connect({"supported": False})


@pytest.mark.parametrize("st, ok", [
    ({"supported": True, "checking": True}, True),                        # not known yet
    ({"supported": True, "error": "no powershell"}, True),                # can't tell: don't nag
    ({"supported": True, "enabled": False, "allowed": False, "blocking_rules": 0}, True),
    ({"supported": True, "enabled": True, "allowed": False, "blocking_rules": 0}, False),
    ({"supported": True, "enabled": True, "allowed": True, "blocking_rules": 2}, False),
    ({"supported": True, "enabled": True, "allowed": True, "blocking_rules": 0}, True),
])
def test_phones_can_connect(st, ok):
    assert phones_can_connect(st) is ok


def test_watcher_rechecks_until_the_user_allows():
    blocked = {"supported": True, "enabled": True, "allowed": False, "blocking_rules": 1}
    allowed = {"supported": True, "enabled": True, "allowed": True, "blocking_rules": 0}
    answers = [blocked, blocked, blocked, allowed]
    checks = []

    def check():
        checks.append(1)
        return answers[min(len(checks), len(answers)) - 1]

    fw = Firewall(check=check, allow=lambda: True, interval=0.01)
    fw.refresh()
    assert wait(fw)["ok"] is False and len(checks) == 1       # plain refresh checks once
    assert fw.request_allow()
    st = wait(fw)
    assert st["ok"] is True and st["allowed"] and len(checks) == 4


def test_declined_prompt_does_not_poll():
    fw = Firewall(check=lambda: pytest.fail("checked"), allow=lambda: False)
    assert fw.request_allow() is False
    assert fw.snapshot()["busy"] is False


def _is_admin() -> bool:
    import ctypes
    return bool(ctypes.windll.shell32.IsUserAnAdmin())


def _ps(script: str) -> str:
    out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", firewall._encode(script)],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return out.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Firewall")
def test_allow_script_replaces_block_rules_on_windows(tmp_path):
    """The script behind "Allow phone connections", run for real (CI runners are admins)."""
    if not _is_admin():
        pytest.skip("needs an administrator shell")
    exe = str(tmp_path / "Fake SplineWire.exe")
    rule = "Spline Wire test rule"
    block = "Spline Wire test block"
    try:
        # What answering Cancel at the firewall prompt leaves behind (path in another case).
        _ps(f"New-NetFirewallRule -DisplayName '{block}' -Direction Inbound -Action Block "
            f"-Program '{exe.upper()}' | Out-Null")
        before = json.loads(_ps(firewall.status_script(exe, rule)))
        assert before["allowed"] is False and before["blocking_rules"] == 1

        _ps(firewall.allow_script(exe, rule))
        after = json.loads(_ps(firewall.status_script(exe, rule)))
        assert after["allowed"] is True and after["blocking_rules"] == 0
        assert phones_can_connect({"supported": True, **after})
    finally:
        _ps(f"Remove-NetFirewallRule -DisplayName '{block}' -ErrorAction SilentlyContinue; "
            f"Remove-NetFirewallRule -DisplayName '{rule}' -ErrorAction SilentlyContinue")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Firewall")
def test_status_reads_on_windows():
    st = firewall.status()
    assert st["supported"] and "error" not in st, st
    assert isinstance(st["networks"], list)
