"""
training/dataset.py

CSA棋譜から共通の局面データセットを生成する。
"""

from __future__ import annotations

import json
from pathlib import Path

from training.csa_parser import (
    build_position_sequence,
    parse_csa_directory,
    parse_csa_file,
)


def normalize_result(result: int) -> int:
    """
    cshogiの勝敗値を研究用ラベルへ変換する。

    注意:
        cshogi.Parserのwin値をここで吸収する。
    """

    # CSA Parserの値については、
    # 実データを増やした段階で必ず検証する。
    #
    # 現段階では値をそのまま保持し、
    # 研究用ラベルへの変換は後で確定する。

    return int(result)


def game_to_records(
    game,
    game_id: int | str = 0,
):
    """
    1局の棋譜を共通局面データへ変換する。
    """

    positions = build_position_sequence(game)

    records = []

    for position in positions:
        records.append(
            {
                "game_id": game_id,
                "ply": position["ply"],
                "sfen": position["sfen"],
                "move": position["move"],
                "result": normalize_result(
                    position["result"]
                ),
            }
        )

    return records


def write_jsonl(
    records,
    output_path: str | Path,
) -> int:
    """JSONLとして保存する。"""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            count += 1

    return count


def convert_csa_to_jsonl(
    csa_path: str | Path,
    output_path: str | Path,
) -> int:
    """1つのCSAファイルをJSONLへ変換する。"""

    games = parse_csa_file(csa_path)

    records = []

    for game_id, game in enumerate(games):
        records.extend(
            game_to_records(
                game,
                game_id=game_id,
            )
        )

    return write_jsonl(
        records,
        output_path,
    )


def convert_directory_to_jsonl(
    csa_directory: str | Path,
    output_path: str | Path,
) -> int:
    """ディレクトリ内の全CSAをJSONLへ変換する。"""

    games = parse_csa_directory(
        csa_directory,
        recursive=True,
    )

    records = []

    for game_id, game in enumerate(games):

        records.extend(
            game_to_records(
                game,
                game_id=game_id,
            )
        )

    return write_jsonl(
        records,
        output_path,
    )