import cshogi

from features.activity import (
    attacks_from_square,
    f06_control,
    f07_mobility,
    f08_important_control,
    f09_idle_improvement,
    f10_activity,
)


def square(name: str) -> int:
    """cshogiのSQUARE_NAMESから盤面座標を取得する。"""
    return cshogi.SQUARE_NAMES.index(name)


def attacks(board, square_name: str):
    """指定マスの駒の利きを座標名の集合で返す。"""
    sq = square(square_name)
    return {
        cshogi.SQUARE_NAMES[s]
        for s in attacks_from_square(board, sq)
    }


def test_pawn_attack_initial_position():
    """初期局面の7gの歩は7fに利く。"""
    board = cshogi.Board()

    assert attacks(board, "7g") == {"7f"}


def test_knight_attack_initial_position():
    """初期局面の8iの桂は7gと9gに利く。"""
    board = cshogi.Board()

    assert attacks(board, "8i") == {"7g", "9g"}


def test_silver_attack_initial_position():
    """初期局面の7iの銀の利きを確認する。"""
    board = cshogi.Board()

    assert attacks(board, "7i") == {
        "6h",
        "7h",
        "8h",
    }


def test_gold_attack_initial_position():
    """初期局面の6iの金の利きを確認する。"""
    board = cshogi.Board()

    assert attacks(board, "6i") == {
        "5h",
        "5i",
        "6h",
        "7h",
        "7i",
    }


def test_king_attack_initial_position():
    """初期局面の5iの玉の利きを確認する。"""
    board = cshogi.Board()

    assert attacks(board, "5i") == {
        "4h",
        "4i",
        "5h",
        "6h",
        "6i",
    }


def test_rook_is_blocked_by_own_piece():
    """
    飛車の前に自駒がある場合、
    その駒より先には利かないことを確認する。
    """
    board = cshogi.Board()

    # 2hの飛車の前には2gの歩がある。
    # したがって2hから2f以遠には利かない。
    rook_attacks = attacks(board, "2h")

    assert "2g" in rook_attacks
    assert "2f" not in rook_attacks
    assert "2e" not in rook_attacks


def test_bishop_is_blocked_by_own_piece():
    """
    角の前に自駒がある場合、
    その駒より先には利かないことを確認する。
    """
    board = cshogi.Board()

    bishop_attacks = attacks(board, "8h")

    # 8hの角から7gには自軍の歩がある。
    assert "7g" in bishop_attacks
    assert "6f" not in bishop_attacks


def test_f06_initial_position_is_balanced():
    """初期局面では黒白の利きが対称になる。"""
    board = cshogi.Board()

    result = f06_control(board)

    assert result.black == result.white
    assert result.difference == 0.0


def test_f07_initial_position_is_balanced():
    """初期局面では黒白の合法手数が等しい。"""
    board = cshogi.Board()

    result = f07_mobility(board)

    assert result.black_legal_moves == result.white_legal_moves
    assert result.difference == 0


def test_f08_initial_position_is_balanced():
    """初期局面では重要地点への利きも黒白で対称になる。"""
    board = cshogi.Board()

    result = f08_important_control(board)

    assert result.black == result.white
    assert result.difference == 0.0


def test_f09_initial_position_returns_valid_result():
    """F09が初期局面で正常に計算できる。"""
    board = cshogi.Board()

    result = f09_idle_improvement(board, board)

    assert result.black_idle == 0
    assert result.white_idle == 0


def test_f10_initial_position_is_balanced():
    """初期局面では活動性の黒白差が0になる。"""
    board = cshogi.Board()

    result = f10_activity(board)

    assert result.black == result.white
    assert result.difference == 0.0
