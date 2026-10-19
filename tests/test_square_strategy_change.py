import cshogi
import pytest

from features.common import BLACK
from interpreter.square_strategy_change import compare_square_strategy


def test_compare_square_strategy_returns_valid_changes():
    before = cshogi.Board()
    after = before.copy()
    after.push_usi("7g7f")

    changes = compare_square_strategy(
        before,
        after,
        BLACK,
        top_n=15,
    )

    assert isinstance(changes, tuple)
    assert len(changes) <= 15

    for item in changes:
        assert 0 <= item.square < 81
        assert item.usi
        assert item.status in ("appeared", "disappeared", "changed")
        assert item.delta == pytest.approx(
            item.after_score - item.before_score
        )


def test_compare_square_strategy_rejects_invalid_top_n():
    board = cshogi.Board()

    with pytest.raises(ValueError):
        compare_square_strategy(
            board,
            board.copy(),
            BLACK,
            top_n=0,
        )
