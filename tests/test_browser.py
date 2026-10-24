"""Tests for the headless-browser rendering fallback (offline)."""

import stat
from pathlib import Path

from slm_pipeline.stage_a.browser import find_chrome, render_page


def test_find_chrome_returns_none_for_unknown_binary():
    assert find_chrome("/nonexistent/browser/definitely-not-here") is None


def test_find_chrome_accepts_absolute_executable_path(tmp_path: Path):
    fake = tmp_path / "mychrome"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    assert find_chrome(str(fake)) == str(fake)


def test_find_chrome_respects_path(monkeypatch, tmp_path: Path):
    fake = tmp_path / "chromium-browser"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert find_chrome() == str(fake)


def test_find_chrome_returns_none_when_path_has_no_browser(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert find_chrome() is None


def test_render_page_returns_empty_without_browser(tmp_path: Path):
    lang_none = tmp_path / "not-a-browser"
    assert render_page("https://example.com", chrome_path=lang_none) == ""


def test_render_page_rejects_non_http_urls(tmp_path: Path):
    # Pass an executable that would be launched if the URL guard failed.
    probe = tmp_path / "probe-browser"
    probe.write_text("#!/bin/sh\nexit 0\n")
    probe.chmod(probe.stat().st_mode | stat.S_IXUSR)
    assert render_page("file:///etc/passwd", chrome_path=str(probe)) == ""
