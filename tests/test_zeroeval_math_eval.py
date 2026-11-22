import json

from eval.chat_benchmarks.zeroeval.src.evaluation.math_eval import eval_model


def test_eval_model_rejects_hedged_prose_mentioning_reference(tmp_path):
    items = [
        {"id": "ok", "output": [json.dumps({"reasoning": "r", "answer": "42"})], "answer": "42"},
        {"id": "units", "output": [json.dumps({"reasoning": "r", "answer": "42 miles"})], "answer": "42"},
        {"id": "exp_unit", "output": [json.dumps({"reasoning": "r", "answer": "42 m^2"})], "answer": "42"},
        {"id": "neg", "output": [json.dumps({"reasoning": "r", "answer": "-3.5"})], "answer": "-3.5"},
        {"id": "wrong", "output": [json.dumps({"reasoning": "r", "answer": "41"})], "answer": "42"},
        {
            "id": "hedged",
            "output": [json.dumps({"reasoning": "r", "answer": "I cannot solve this; the answer is not 42"})],
            "answer": "42",
        },
        {"id": "prose_after", "output": [json.dumps({"reasoning": "r", "answer": "42 is wrong"})], "answer": "42"},
    ]
    fp = tmp_path / "results.json"
    fp.write_text(json.dumps(items), encoding="utf-8")

    _, parsed = eval_model("dummy%greedy", str(fp))
    matched = {row["id"]: row["matched"] for row in parsed}

    assert matched == {
        "ok": True,
        "units": True,
        "exp_unit": True,
        "neg": True,
        "wrong": False,
        "hedged": False,
        "prose_after": False,
    }
