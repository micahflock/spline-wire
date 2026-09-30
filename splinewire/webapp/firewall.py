"""Windows Firewall: let the phone reach the app.

Windows drops incoming connections to a new program until the user answers
the firewall prompt, and answering "Cancel" adds a rule that blocks the
program. Either way the phone's request just times out. This module
reports whether phones can connect and, with one UAC prompt, fixes it:

- removes inbound *block* rules for this program (block rules win over
  allow rules), and
- adds one inbound allow rule for the app's TCP ports, not tied to the
  program path, so it keeps working when a newer SplineWire.exe is
  downloaded somewhere else.

The app still needs its secret token for anything beyond the page itself.
Everything here is a no-op off Windows.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from typing import Callable

RULE_NAME = "Spline Wire (phone upload)"
PORTS = "8765-8774"   # DEFAULT_PORT and the fallbacks start_server tries

_CREATE_NO_WINDOW = 0x08000000


def _ps_quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def status_script(exe: str, rule: str = RULE_NAME) -> str:
    # Rules from the firewall prompt may spell the path with different case
    # or environment variables, so compare expanded paths ignoring case.
    return f"""
$ErrorActionPreference = 'SilentlyContinue'
$exe = {_ps_quote(exe)}
$on = @(Get-NetFirewallProfile | Where-Object {{ $_.Enabled -eq 'True' }}).Count -gt 0
$rule = @(Get-NetFirewallRule -DisplayName {_ps_quote(rule)} | Where-Object {{ $_.Enabled -eq 'True' -and $_.Action -eq 'Allow' }}).Count -gt 0
$blocks = @(Get-NetFirewallApplicationFilter |
  Where-Object {{ $_.Program -and [Environment]::ExpandEnvironmentVariables($_.Program) -ieq $exe }} |
  Get-NetFirewallRule |
  Where-Object {{ $_.Enabled -eq 'True' -and $_.Action -eq 'Block' -and $_.Direction -eq 'Inbound' }}).Count
$nets = @(Get-NetConnectionProfile | ForEach-Object {{ @{{ name = [string]$_.Name; category = [string]$_.NetworkCategory; adapter = [string]$_.InterfaceAlias }} }})
@{{ enabled = $on; allowed = $rule; blocking_rules = $blocks; networks = $nets }} | ConvertTo-Json -Compress -Depth 4
"""


def allow_script(exe: str, rule: str = RULE_NAME) -> str:
    return f"""
$ErrorActionPreference = 'Stop'
$exe = {_ps_quote(exe)}
Get-NetFirewallApplicationFilter |
  Where-Object {{ $_.Program -and [Environment]::ExpandEnvironmentVariables($_.Program) -ieq $exe }} |
  Get-NetFirewallRule |
  Where-Object {{ $_.Action -eq 'Block' -and $_.Direction -eq 'Inbound' }} | Remove-NetFirewallRule
Remove-NetFirewallRule -DisplayName {_ps_quote(rule)} -ErrorAction SilentlyContinue
New-NetFirewallRule -DisplayName {_ps_quote(rule)} -Direction Inbound -Action Allow -Protocol TCP `
  -LocalPort {PORTS} -Profile Any -Description 'Lets a phone on the same network upload photos to Spline Wire.' | Out-Null
"""


def status() -> dict:
    """{"supported": False} off Windows; otherwise whether phones can connect.

    enabled: the firewall is on for some network type. allowed: our allow
    rule exists. blocking_rules: inbound block rules for
    this program (from answering Cancel to the firewall prompt), which win
    over any allow rule. networks: name and category (Public/Private) of
    each connected network.
    """
    if sys.platform != "win32":
        return {"supported": False}
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", _encode(status_script(sys.executable))],
            capture_output=True, text=True, timeout=60, creationflags=_CREATE_NO_WINDOW,
        )
        if out.returncode or not out.stdout.strip():
            raise ValueError(out.stderr.strip()[-300:] or f"powershell exit code {out.returncode}")
        data = json.loads(out.stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired) as err:
        return {"supported": True, "error": f"couldn't read firewall settings: {err}"}
    nets = data.get("networks") or []
    if isinstance(nets, dict):     # ConvertTo-Json unwraps a one-element array
        nets = [nets]
    return {
        "supported": True,
        "enabled": bool(data.get("enabled", True)),
        "allowed": bool(data.get("allowed")),
        "blocking_rules": int(data.get("blocking_rules") or 0),
        "networks": nets,
    }


def request_allow() -> bool:
    """Run allow_script elevated (shows the UAC prompt). True if it was launched;
    the caller re-checks status() to see whether the user agreed."""
    if sys.platform != "win32":
        return False
    import ctypes
    # EncodedCommand avoids every quoting problem between ShellExecute and PowerShell.
    args = f"-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand {_encode(allow_script(sys.executable))}"
    result = ctypes.windll.shell32.ShellExecuteW(None, "runas", "powershell", args, None, 0)
    return int(result) > 32


def _encode(script: str) -> str:
    import base64
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


def phones_can_connect(st: dict) -> bool:
    """False when the firewall is known to stop phones reaching the app."""
    if not st.get("supported") or st.get("error") or "allowed" not in st:
        return True
    if not st.get("enabled", True):
        return True
    return bool(st["allowed"]) and not st.get("blocking_rules")


class Firewall:
    """status() cached for the pages, refreshed in the background (it takes
    PowerShell a few seconds)."""

    def __init__(self, check: Callable[[], dict] = status,
                 allow: Callable[[], bool] = request_allow, interval: float = 2.0) -> None:
        self._check, self._allow, self._interval = check, allow, interval
        self._lock = threading.Lock()
        self._status: dict = {"supported": sys.platform == "win32", "checking": True}
        self._busy = False
        self._fix_until = 0.0

    def snapshot(self) -> dict:
        with self._lock:
            out = dict(self._status)
            out["busy"] = self._busy
        out["ok"] = phones_can_connect(out)
        return out

    def refresh(self, wait_for_fix: float = 0.0) -> None:
        """Re-check in the background. With wait_for_fix, keep re-checking for
        that many seconds until phones can connect (the user is answering the
        Windows prompt meanwhile)."""
        with self._lock:
            self._fix_until = max(self._fix_until, time.monotonic() + wait_for_fix)
            if self._busy:
                return        # the running check picks up the new deadline
            self._busy = True
        threading.Thread(target=self._run, daemon=True, name="splinewire-firewall").start()

    def request_allow(self) -> bool:
        launched = self._allow()
        if launched:
            self.refresh(wait_for_fix=120.0)
        return launched

    def _run(self) -> None:
        try:
            while True:
                st = self._check()
                with self._lock:
                    self._status = st
                    if phones_can_connect(st) or time.monotonic() >= self._fix_until:
                        self._busy = False
                        return
                time.sleep(self._interval)
        except BaseException:
            with self._lock:
                self._busy = False
            raise
