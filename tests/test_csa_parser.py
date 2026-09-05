"""
tests/test_csa_parser.py

CSA棋譜読み込み処理のテスト。
"""

from pathlib import Path

import cshogi

from training.csa_parser import (
    CSAGame,
    build_position_sequence,
    parse_csa_directory,
    parse_csa_file,
)


TEST_CSA = """\
V2.2
N+TEST_BLACK
N-TEST_WHITE
$EVENT:TEST
PI
+
+7776FU
-3334FU
+2726FU
-8384FU
+2625FU
-8485FU
+7978GI
-4132KI
+8877KA
-5142OU
%TORYO
"""


def create_test_csa(tmp_path: Path) -> Path:
    """テスト用CSAを作成する。"""

    path = tmp_path / "test.csa"
    path.write_text(
        TEST_CSA,
        encoding="utf-8",
    )

    return path


def test_parse_csa_file(tmp_path):
    """CSAを正常に読み込める。"""

    path = create_test_csa(tmp_path)

    games = parse_csa_file(path)

    assert len(games) == 1

    game = games[0]

    assert isinstance(game, CSAGame)
    assert game.move_count == 10
    assert game.sfen.startswith(
        "lnsgkgsnl/"
    )
    assert game.names == [
        "TEST_BLACK",
        "TEST_WHITE",
    ]
    assert game.endgame == "%TORYO"


def test_csa_moves_are_legal(tmp_path):
    """読み込んだ全指し手が合法である。"""

    path = create_test_csa(tmp_path)

    games = parse_csa_file(path)
    game = games[0]

    board = cshogi.Board(sfen=game.sfen)

    for move in game.moves:
        assert board.is_legal(move)
        board.push(move)


def test_build_position_sequence(tmp_path):
    """各指し手直前の局面を生成できる。"""

    path = create_test_csa(tmp_path)

    games = parse_csa_file(path)
    game = games[0]

    positions = build_position_sequence(game)

    assert len(positions) == 10

    assert positions[0]["ply"] == 1
    assert positions[0]["move"] == "7g7f"

    assert positions[1]["ply"] == 2
    assert positions[1]["move"] == "3c3d"

    assert positions[-1]["ply"] == 10
    assert positions[-1]["move"] == "5a4b"


def test_position_sequence_contains_sfen(tmp_path):
    """各局面にSFENが保存されている。"""

    path = create_test_csa(tmp_path)

    games = parse_csa_file(path)
    positions = build_position_sequence(games[0])

    for position in positions:
        assert isinstance(position["sfen"], str)
        assert position["sfen"]


def test_parse_csa_directory(tmp_path):
    """ディレクトリ内のCSAをまとめて読み込める。"""

    csa_dir = tmp_path / "csa"
    csa_dir.mkdir()

    (csa_dir / "game1.csa").write_text(
        TEST_CSA,
        encoding="utf-8",
    )

    (csa_dir / "game2.csa").write_text(
        TEST_CSA,
        encoding="utf-8",
    )

    games = parse_csa_directory(csa_dir)

    assert len(games) == 2
    assert all(
        game.move_count == 10
        for game in games
    )


def test_parse_csa_file_not_found():
    """存在しないCSAはFileNotFoundError。"""

    missing = Path(
        "data/raw/does_not_exist.csa"
    )

    try:
        parse_csa_file(missing)
        assert False
    except FileNotFoundError:
        pass