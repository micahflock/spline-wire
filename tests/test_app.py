"""splinewire-app, the program SplineWire.exe runs: start, serve, quit."""
import json
import threading
import time
import urllib.request

from splinewire.settings import settings_path
from splinewire.webapp import app as app_module


def _post(url: str, token: str) -> dict:
    req = urllib.request.Request(url, data=b"", method="POST", headers={"X-Token": token})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)


def test_the_app_serves_until_quit_and_a_second_copy_defers_to_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))        # where its settings go, on any system
    opened, servers = [], []
    monkeypatch.setattr(app_module.webbrowser, "open", opened.append)
    start = app_module.start_server
    monkeypatch.setattr(app_module, "start_server", lambda *a, **k: servers.append(start(*a, **k)) or servers[-1])

    exit_codes = []
    running = threading.Thread(target=lambda: exit_codes.append(
        app_module.main(["--port", "0", "--workspace", str(tmp_path / "ws")])))
    running.start()
    deadline = time.monotonic() + 20
    while not (servers and opened) and time.monotonic() < deadline:
        time.sleep(0.02)
    server = servers[0]
    try:
        # It opened the desktop page with its token, and saved that token for next time.
        assert opened == [server.desktop_url()] and server.app.token in opened[0]
        assert json.loads(settings_path().read_text(encoding="utf-8"))["token"] == server.app.token
        assert settings_path().is_relative_to(tmp_path)
        assert server.app.workspace == tmp_path / "ws"
        said = capsys.readouterr().out
        assert f"http://localhost:{server.port}/" in said and "Keep this window open" in said

        # Started again on the same port, it finds the first copy and opens that instead.
        assert app_module._already_running(server.port)
        assert app_module.main(["--port", str(server.port)]) == 0
        assert opened[-1] == f"http://localhost:{server.port}/" and len(servers) == 1
        assert "already running" in capsys.readouterr().out
    finally:
        _post(f"http://127.0.0.1:{server.port}/api/quit", server.app.token)     # the page's Quit button
        running.join(20)
        server.server_close()
    assert exit_codes == [0] and not running.is_alive()
    assert not app_module._already_running(server.port)


def test_no_copy_is_running_on_a_port_nothing_listens_on():
    assert app_module._already_running(1) is False
