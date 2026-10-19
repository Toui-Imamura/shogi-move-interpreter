import cshogi

from features.activity import all_attacks
from features.common import BLACK, WHITE, opponent, piece_color
from interpreter.square_strategy import analyze_square_candidates


def test_attack_candidates_do_not_include_unattacked_enemy_pieces():
    board = cshogi.Board()
    color = BLACK
    enemy = opponent(color)

    own_attacks = all_attacks(board, color)
    attacked_squares = {
        target
        for targets in own_attacks.values()
        for target in targets
    }

    attack_candidates, _ = analyze_square_candidates(
        board,
        color=color,
        top_n=81,
    )

    for candidate in attack_candidates:
        piece = board.piece(candidate.square)

        if piece != 0 and piece_color(piece) == enemy:
            assert candidate.square in attacked_squares


def test_square_strategy_returns_separate_candidate_lists():
    board = cshogi.Board()

    attacks, defenses = analyze_square_candidates(
        board,
        color=BLACK,
        top_n=5,
    )

    assert isinstance(attacks, list)
    assert isinstance(defenses, list)
    assert len(attacks) <= 5
    assert len(defenses) <= 5


def test_square_strategy_rejects_invalid_top_n():
    board = cshogi.Board()

    try:
        analyze_square_candidates(board, color=BLACK, top_n=0)
    except ValueError:
        pass
    else:
        raise AssertionError("top_n=0 should raise ValueError")


def test_square_strategy_accepts_both_sides():
    board = cshogi.Board()

    for color in (BLACK, WHITE):
        attacks, defenses = analyze_square_candidates(
            board,
            color=color,
            top_n=3,
        )
        assert len(attacks) <= 3
        assert len(defenses) <= 3


def test_strategy_separates_targets_bases_and_defenses():
    from interpreter.square_strategy import analyze_square_strategy

    board = cshogi.Board()
    targets, bases, defenses = analyze_square_strategy(
        board,
        color=BLACK,
        top_n=81,
    )

    # 攻撃対象は敵駒、攻めの拠点は空きマスであること。
    for candidate in targets:
        piece = board.piece(candidate.square)
        assert piece != cshogi.NONE
        assert piece_color(piece) == WHITE
        assert candidate.kind == "attack_target"

    for candidate in bases:
        assert board.piece(candidate.square) == cshogi.NONE
        assert candidate.kind == "attack_base"

    for candidate in defenses:
        assert candidate.kind in ("defense_piece", "defense_point")


def test_strategy_rejects_invalid_top_n():
    from interpreter.square_strategy import analyze_square_strategy

    board = cshogi.Board()

    try:
        analyze_square_strategy(board, color=BLACK, top_n=0)
    except ValueError:
        pass
    else:
        raise AssertionError("top_n=0 should raise ValueError")
