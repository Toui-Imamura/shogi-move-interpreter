from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import cshogi
import torch
from torch.utils.data import Dataset


BLACK = cshogi.BLACK
WHITE = cshogi.WHITE

# 教師評価値を [-1, 1] に正規化するスケール。
# 1000cp -> tanh(1) ≈ 0.762
TEACHER_SCORE_SCALE = 1000.0


@dataclass(frozen=True)
class NNUERecord:
    """
    NNUE学習用の1局面。

    sfen:
        学習対象局面

    teacher_score_cp_black:
        教師エンジンによるBlack視点の評価値(cp)

    ply:
        手数

    side_to_move:
        局面で手番になっている側
    """

    sfen: str
    teacher_score_cp_black: float
    ply: int
    side_to_move: int


def load_position_records(
    path: str | Path,
) -> list[NNUERecord]:
    """
    teacher_values形式のJSONLからNNUE学習用レコードを読み込む。
    """

    records: list[NNUERecord] = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            data = json.loads(line)

            if "sfen" not in data:
                raise ValueError(
                    f"Missing 'sfen' at line {line_number}"
                )

            if "teacher_score_cp_black" not in data:
                raise ValueError(
                    "Missing 'teacher_score_cp_black' "
                    f"at line {line_number}"
                )

            board = cshogi.Board(
                sfen=data["sfen"]
            )

            records.append(
                NNUERecord(
                    sfen=data["sfen"],
                    teacher_score_cp_black=float(
                        data["teacher_score_cp_black"]
                    ),
                    ply=int(data["ply"]),
                    side_to_move=int(board.turn),
                )
            )

    return records


def teacher_score_to_target(
    teacher_score_cp_black: float,
) -> float:
    """
    Black視点の教師評価値(cp)をNNUEの教師値[-1, 1]へ変換する。

    変換:
        target = tanh(cp / TEACHER_SCORE_SCALE)

    例:
        0cp       -> 0.000
        500cp     -> 約0.462
        1000cp    -> 約0.762
        2000cp    -> 約0.964
        100000cp  -> 1.000
        -100000cp -> -1.000
    """

    return math.tanh(
        teacher_score_cp_black / TEACHER_SCORE_SCALE
    )


class NNUEPositionDataset(Dataset):
    """
    teacher_values形式JSONLからNNUE学習用の局面を提供するDataset。
    """

    def __init__(
        self,
        path: str | Path,
    ):
        self.records = load_position_records(
            path
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self,
        index: int,
    ) -> dict:

        record = self.records[index]

        target = teacher_score_to_target(
            record.teacher_score_cp_black
        )

        return {
            "sfen": record.sfen,
            "ply": record.ply,
            "side_to_move": record.side_to_move,
            "teacher_score_cp_black": record.teacher_score_cp_black,
            "target": torch.tensor(
                target,
                dtype=torch.float32,
            ),
        }
