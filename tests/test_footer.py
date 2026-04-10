from holofile._footer import HoloFooter


def test_round_trip():
    data = {"key": "value", "number": 42, "nested": {"a": 1}}
    f = HoloFooter(data=data)
    f2 = HoloFooter.from_json(f.to_json())
    assert f2.data == data


def test_empty():
    f = HoloFooter.empty()
    assert f.data == {}


def test_get_existing():
    f = HoloFooter(data={"fps": 30})
    assert f.get("fps") == 30


def test_get_missing_default():
    f = HoloFooter(data={})
    assert f.get("missing", 99) == 99


def test_unknown_keys_preserved():
    raw = '{"future_key": true, "version": 99}'
    f = HoloFooter.from_json(raw)
    assert f.get("future_key") is True
    assert f.get("version") == 99
