from interpreter.explanation_rules import (
    explain_from_deltas,
    generate_explanation_candidates,
    select_top_explanation_candidates,
)


def test_generate_attack_candidate() -> None:
    deltas = {
        "F11": 0.8,
    }

    candidates = generate_explanation_candidates(deltas)

    assert any(
        candidate.feature_name == "F11"
        for candidate in candidates
    )


def test_select_top_candidates() -> None:
    deltas = {
        "F11": 0.8,
        "F24": 0.7,
        "F28": 0.5,
    }

    candidates = generate_explanation_candidates(deltas)
    selected = select_top_explanation_candidates(
        candidates,
        max_candidates=2,
    )

    assert len(selected) <= 2


def test_explain_from_deltas() -> None:
    explanation = explain_from_deltas(
        move_usi="7g7f",
        deltas={
            "F11": 0.8,
        },
    )

    assert "7g7f" in explanation
    assert "攻撃" in explanation
