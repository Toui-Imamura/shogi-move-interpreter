import json

from analysis.analyze_feature_distribution import (
    analyze_feature_distribution,
    percentile,
    summarize_feature,
)


def test_percentile():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert percentile(values, 0) == 1.0
    assert percentile(values, 50) == 3.0
    assert percentile(values, 100) == 5.0


def test_summarize_feature():
    values = [
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
    ]

    summary = summarize_feature(values)

    assert summary["count"] == 5
    assert summary["mean"] == 0.0
    assert summary["min"] == -2.0
    assert summary["median"] == 0.0
    assert summary["max"] == 2.0


def test_analyze_feature_distribution(
    tmp_path,
):
    input_path = (
        tmp_path / "transition.jsonl"
    )

    records = [
        {
            "feature_deltas": {
                "F01": 0.0,
                "F06": 3.0,
            }
        },
        {
            "feature_deltas": {
                "F01": 1.0,
                "F06": -3.0,
            }
        },
        {
            "feature_deltas": {
                "F01": -1.0,
                "F06": 6.0,
            }
        },
    ]

    with input_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for record in records:
            f.write(
                json.dumps(record)
                + "\n"
            )

    summary = analyze_feature_distribution(
        input_path
    )

    assert set(summary) == {
        "F01",
        "F06",
    }

    assert summary["F01"]["count"] == 3
    assert summary["F01"]["mean"] == 0.0

    assert summary["F06"]["count"] == 3
    assert summary["F06"]["min"] == -3.0
    assert summary["F06"]["max"] == 6.0
