"""Per-user settings: a convenience that must never stop the app."""
import json
from pathlib import Path

from splinewire.settings import default_workspace, load_settings, save_settings, settings_path


def test_settings_round_trip(tmp_path):
    file = tmp_path / "not" / "there" / "yet" / "settings.json"
    assert load_settings(file) == {}
    save_settings({"token": "abc", "chain": {"pitch_mm": 10.0}}, file)
    assert load_settings(file) == {"token": "abc", "chain": {"pitch_mm": 10.0}}
    save_settings({"token": "abc", "chain": {"pitch_mm": 12.5}}, file)
    assert load_settings(file)["chain"] == {"pitch_mm": 12.5}
    # Written beside the file and swapped in whole: nothing half-written is left behind.
    assert [p.name for p in file.parent.iterdir()] == ["settings.json"]


def test_a_file_that_is_not_settings_reads_as_none(tmp_path):
    file = tmp_path / "settings.json"
    for damaged in ('{"token": "ab', "", "[1, 2]", "null", '"text"', "\x00\x00"):
        file.write_text(damaged, encoding="utf-8")
        assert load_settings(file) == {}


def test_settings_that_cannot_be_saved_are_let_go(tmp_path):
    in_the_way = tmp_path / "a-file"
    in_the_way.write_text("x")
    save_settings({"token": "abc"}, in_the_way / "settings.json")       # its folder can't be made: no error
    assert in_the_way.read_text() == "x"


def test_where_settings_and_photos_live(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert settings_path() == tmp_path / "Roaming" / "SplineWire" / "settings.json"
    monkeypatch.delenv("APPDATA")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert settings_path() == tmp_path / ".config" / "splinewire" / "settings.json"
    assert default_workspace() == tmp_path / "SplineWire"               # no Documents folder here
    (tmp_path / "Documents").mkdir()
    assert default_workspace() == tmp_path / "Documents" / "SplineWire"
    save_settings({"side": "outside"})
    assert json.loads(settings_path().read_text(encoding="utf-8")) == {"side": "outside"}
