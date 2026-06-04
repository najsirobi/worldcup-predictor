"""Test World Bank fetch pagination (no network: requests.get is monkeypatched)."""
import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).parent.parent / "scripts" / "fetch_world_bank_country_context.py"
_spec = importlib.util.spec_from_file_location("fetch_wb_fetch", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_fetch_indicator_follows_all_pages(monkeypatch):
    """All pages are concatenated, driven by the 'pages' metadata."""
    pages = {
        "1": [{"page": 1, "pages": 3, "per_page": 2}, [{"countryiso3code": "AFE", "date": "2020", "value": 1}]],
        "2": [{"page": 2, "pages": 3, "per_page": 2}, [{"countryiso3code": "FRA", "date": "2020", "value": 2}]],
        "3": [{"page": 3, "pages": 3, "per_page": 2}, [{"countryiso3code": "BRA", "date": "2020", "value": 3}]],
    }

    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params["page"])
        return _FakeResponse(pages[params["page"]])

    monkeypatch.setattr(_mod.requests, "get", fake_get)
    monkeypatch.setattr(_mod.time, "sleep", lambda *_a, **_k: None)

    records = _mod.fetch_indicator("SP.POP.TOTL")

    assert calls == ["1", "2", "3"]  # stopped exactly at pages=3, no over-fetch
    assert [r["countryiso3code"] for r in records] == ["AFE", "FRA", "BRA"]


def test_fetch_indicator_single_page(monkeypatch):
    payload = [{"page": 1, "pages": 1}, [{"countryiso3code": "FRA", "date": "2020", "value": 9}]]
    monkeypatch.setattr(_mod.requests, "get", lambda url, params=None, timeout=None: _FakeResponse(payload))

    records = _mod.fetch_indicator("SP.POP.TOTL")
    assert len(records) == 1


def test_fetch_indicator_no_data(monkeypatch):
    payload = [{"message": [{"id": "120", "value": "no data"}]}]
    monkeypatch.setattr(_mod.requests, "get", lambda url, params=None, timeout=None: _FakeResponse(payload))

    assert _mod.fetch_indicator("BAD.CODE") is None
