import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("qs", Path(__file__).resolve().parent.parent / "scripts" / "quality_suite.py")
qs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qs)


def test_json_grader():
    assert qs.grade_json('Sure: {"name": "Ana", "age": 31}', {"name": str, "age": int})
    assert not qs.grade_json('{"name": "Ana", "age": "31"}', {"name": str, "age": int})
    assert not qs.grade_json('{"ok": 1}', {"ok": bool})
    assert not qs.grade_json('{"age": true}', {"age": int})  # bool is not an int here
    assert not qs.grade_json("no json", {"a": int})


def test_arith_grader():
    assert qs.grade_arith("The answer is 54.", 54)
    assert qs.grade_arith("1,000", 1000)
    assert not qs.grade_arith("It is 53 or 54", 54)  # first number counts
    assert not qs.grade_arith("fifty-four", 54)


def test_minimums_file_is_complete():
    import json

    m = json.loads((Path(__file__).resolve().parent.parent / "docs" / "quality_minimums.json").read_text())
    assert {"json", "arith", "follow"} <= set(m)
