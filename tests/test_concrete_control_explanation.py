from __future__ import annotations

import cshogi

from interpreter.concrete_control_explanation import (
    build_control_sentence,
    extract_concrete_control_evidence,
)


def test_extract_control_change_after_pawn_move():
    before = cshogi.Board()
    move = before.move_from_usi("7g7f")

    after = before.copy()
    after.push(move)

    evidence = extract_concrete_control_evidence(before, after)

    assert evidence

    bishop_change = [
        item
        for item in evidence
        if item.piece_square == "8h"
        and item.piece_name == "角"
    ]

    assert bishop_change

    added = bishop_change[0].added_squares

    assert "3c" in added
    assert "4d" in added
    assert "5e" in added
    assert "6f" in added


def test_build_control_sentence():
    evidence = (
        type(
            "Evidence",
            (),
            {
                "color": cshogi.BLACK,
                "piece_square": "8h",
                "piece_name": "角",
                "added_squares": ("3c", "4d", "5e", "6f"),
                "removed_squares": (),
            },
        )(),
    )

    sentence = build_control_sentence(evidence)

    assert "8hの角" in sentence
    assert "3c" in sentence
    assert "4d" in sentence
    assert "5e" in sentence
    assert "6f" in sentence
