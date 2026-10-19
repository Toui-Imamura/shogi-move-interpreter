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

TEACHER_SCORE_SCALE = 1000.0


@dataclass(frozen=True)
class NNUERecord:
    """NNUE学習用の1局面。"""

    sfen: str
    teacher_score_cp_black: float
    ply: int
    side_to_move: int
    game_id: str
    teacher_mate: int | None = None


def load_position_records(
    path: str | Path,
    exclude_mate: bool = False,
) -> list[NNUERecord]:
    """
    teacher_values形式のJSONLを読み込む。

    exclude_mate=True の場合、teacher_mate が None でない
    レコードを除外する。元データは変更しない。
    """

    records: list[NNUERecord] = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}"
                ) from exc

            required = (
                "sfen",
                "teacher_score_cp_black",
                "ply",
                "game_id",
            )
            for key in required:
                if key not in data:
                    raise ValueError(
                        f"Missing {key!r} at line {line_number}"
                    )

            mate_value = data.get("teacher_mate")

            if exclude_mate and mate_value is not None:
                continue

            score = float(data["teacher_score_cp_black"])
            if not math.isfinite(score):
                raise ValueError(
                    f"Non-finite teacher score at line {line_number}"
                )

            board = cshogi.Board(sfen=data["sfen"])

            records.append(
                NNUERecord(
                    sfen=data["sfen"],
                    teacher_score_cp_black=score,
                    ply=int(data["ply"]),
                    side_to_move=int(board.turn),
                    game_id=str(data["game_id"]),
                    teacher_mate=(
                        None if mate_value is None
                        else int(mate_value)
                    ),
                )
            )

    return records


def teacher_score_to_target(
    teacher_score_cp_black: float,
) -> float:
    """黒側視点の評価値(cp)を [-1, 1] に正規化する。"""

    return math.tanh(
        teacher_score_cp_black / TEACHER_SCORE_SCALE
    )


class NNUEPositionDataset(Dataset):
    """教師評価値JSONLからNNUE学習用の局面を提供する。"""

    def __init__(
        self,
        path: str | Path,
        exclude_mate: bool = False,
    ):
        self.records = load_position_records(
            path,
            exclude_mate=exclude_mate,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]

        target = teacher_score_to_target(
            record.teacher_score_cp_black
        )

        return {
            "sfen": record.sfen,
            "ply": record.ply,
            "side_to_move": record.side_to_move,
            "game_id": record.game_id,
            "teacher_mate": record.teacher_mate,
            "teacher_score_cp_black": (
                record.teacher_score_cp_black
            ),
            "target": torch.tensor(
                target,
                dtype=torch.float32,
            ),
        }
