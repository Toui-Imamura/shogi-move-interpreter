"""
training/csa_parser.py

CSA棋譜をcshogiで読み込み、
学習データ生成に利用できる形式へ変換する。

想定環境:
    Python 3.12.3
    cshogi 1.0.4
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cshogi
from cshogi import CSA


@dataclass(frozen=True)
class CSAGame:
    """1局分のCSA棋譜。"""

    path: str
    sfen: str
    moves: List[int]
    win: int
    endgame: str
    names: List[str]
    ratings: List[int]
    times: List[int]
    comments: List[str]

    @property
    def move_count(self) -> int:
        """棋譜の手数。"""
        return len(self.moves)


def parse_csa_file(path: str | Path) -> List[CSAGame]:
    """
    CSAファイルを読み込む。

    cshogi.CSA.Parser.parse_file() は
    1ファイルから複数棋譜を返せるため、
    List[CSAGame] とする。
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"CSA file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Not a file: {path}"
        )

    try:
        records = CSA.Parser.parse_file(str(path))
    except Exception as exc:
        raise ValueError(
            f"Failed to parse CSA file: {path}"
        ) from exc

    games: List[CSAGame] = []

    for record in records:
        games.append(
            CSAGame(
                path=str(path),
                sfen=record.sfen,
                moves=list(record.moves),
                win=record.win,
                endgame=record.endgame,
                names=list(record.names),
                ratings=list(record.ratings),
                times=list(record.times),
                comments=list(record.comments),
            )
        )

    return games


def parse_csa_directory(
    directory: str | Path,
    recursive: bool = True,
) -> List[CSAGame]:
    """
    ディレクトリ内のCSA棋譜をまとめて読み込む。
    """

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"CSA directory not found: {directory}"
        )

    if not directory.is_dir():
        raise ValueError(
            f"Not a directory: {directory}"
        )

    if recursive:
        paths = sorted(directory.rglob("*.csa"))
    else:
        paths = sorted(directory.glob("*.csa"))

    games: List[CSAGame] = []

    for path in paths:
        try:
            games.extend(parse_csa_file(path))
        except (ValueError, FileNotFoundError) as exc:
            print(f"[WARNING] {exc}")

    return games


def move_to_usi(move: int) -> str:
    """
    cshogi内部のmoveをUSI形式へ変換する。
    """

    return cshogi.move_to_usi(move)


def build_position_sequence(
    game: CSAGame,
) -> List[dict]:
    """
    1局のCSA棋譜から各局面を生成する。

    各レコードは、

        局面
        ↓
        実際に指された手

    を表す。

    例:

        {
            "ply": 1,
            "sfen": "...",
            "move": "7g7f",
            "result": 1
        }
    """

    board = cshogi.Board(sfen=game.sfen)

    positions: List[dict] = []

    for ply, move in enumerate(game.moves, start=1):

        if not board.is_legal(move):
            raise ValueError(
                f"Illegal move at ply {ply}: "
                f"{move_to_usi(move)}"
            )

        positions.append(
            {
                "ply": ply,
                "sfen": board.sfen(),
                "move": move_to_usi(move),
                "result": game.win,
                "endgame": game.endgame,
                "source": game.path,
            }
        )

        board.push(move)

    return positions