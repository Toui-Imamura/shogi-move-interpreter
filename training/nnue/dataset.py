from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import cshogi
import torch
from torch.utils.data import Dataset


BLACK = cshogi.BLACK
WHITE = cshogi.WHITE


@dataclass(frozen=True)
class NNUERecord:
    """
    NNUE学習用の1局面。

    sfen:
        学習対象局面

    result:
        CSA Parserから取得した勝敗値

    ply:
        手数

    side_to_move:
        局面で手番になっている側
    """

    sfen: str
    result: int
    ply: int
    side_to_move: int


def load_position_records(
    path: str | Path,
) -> list[NNUERecord]:

    records: list[NNUERecord] = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            board = cshogi.Board(
                sfen=data["sfen"]
            )

            records.append(
                NNUERecord(
                    sfen=data["sfen"],
                    result=int(data["result"]),
                    ply=int(data["ply"]),
                    side_to_move=int(board.turn),
                )
            )

    return records


def result_to_target(
    result: int,
    side_to_move: int,
) -> float:
    """
    ゲーム結果を[-1, 1]の教師値へ変換する。

    target:
        +1.0 : 手番側の勝利
        -1.0 : 手番側の敗北
         0.0 : 引き分け

    注意:
        CSAの勝敗定数はcshogiの定数を使用する。
    """

    if result == cshogi.DRAW:
        return 0.0

    if result == cshogi.BLACK_WIN:
        return 1.0 if side_to_move == BLACK else -1.0

    if result == cshogi.WHITE_WIN:
        return 1.0 if side_to_move == WHITE else -1.0

    raise ValueError(
        f"Unknown CSA result value: {result}"
    )


class NNUEPositionDataset(Dataset):
    """
    positions.jsonlからNNUE学習用の局面を提供するDataset。
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

        target = result_to_target(
            record.result,
            record.side_to_move,
        )

        return {
            "sfen": record.sfen,
            "ply": record.ply,
            "side_to_move": record.side_to_move,
            "target": torch.tensor(
                target,
                dtype=torch.float32,
            ),
        }
