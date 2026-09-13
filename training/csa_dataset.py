from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cshogi


def collect_csa_files(input_dir: str | Path) -> list[Path]:
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"input directory not found: {input_dir}"
        )

    files = sorted(input_dir.rglob("*.csa"))

    if not files:
        raise FileNotFoundError(
            f"no CSA files found under: {input_dir}"
        )

    return files


def read_csa_text(csa_path: Path) -> list[str]:
    """
    CSAファイルを読み込む。

    WCSC36は
        'CSA encoding=SHIFT_JIS
    で始まるため、CP932で読む。
    """

    data = csa_path.read_bytes()

    # WCSC36はShift_JIS。
    # PythonではCP932で読むことで実用上ほぼ同等に扱える。
    try:
        text = data.decode("cp932")
    except UnicodeDecodeError:
        # 念のためUTF-8も試す。
        text = data.decode("utf-8")

    return text.splitlines()


def parse_csa_file(
    csa_path: Path,
    game_id: int,
) -> list[dict[str, Any]]:
    """
    CSA V2/V3の棋譜から、

        局面
        指し手
        最終結果

    を抽出する。

    CSA V3の評価値/PVコメント('**)は、
    指し手ではないため、この段階では無視する。
    """

    lines = read_csa_text(csa_path)

    board = cshogi.Board()

    records: list[dict[str, Any]] = []

    result = None
    started = False
    ply = 0

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # --------------------------------------------------
        # 初期盤面情報
        # --------------------------------------------------

        if line.startswith("P"):
            continue

        # --------------------------------------------------
        # 手番開始
        # --------------------------------------------------

        if line == "+" or line == "-":
            started = True

            if line == "-":
                board.turn = cshogi.WHITE

            continue

        if not started:
            continue

        # --------------------------------------------------
        # コメント・評価値・PV
        # --------------------------------------------------

        if line.startswith("'"):
            continue

        # --------------------------------------------------
        # 特殊終局手
        # --------------------------------------------------

        if line.startswith("%"):
            result = line
            break

        # --------------------------------------------------
        # CSAの通常指し手
        #
        # 例:
        # +7776FU,T0
        # -8384FU,T0
        # --------------------------------------------------

        if line[0] not in "+-":
            continue

        if len(line) < 7:
            continue

        color = line[0]

        move_part = line[1:]

        # カンマ以降は時間情報なので除去
        csa_move = move_part.split(",", 1)[0]

        # CSA指し手は6文字
        if len(csa_move) != 6:
            continue

        # --------------------------------------------------
        # 指し手をcshogiで変換
        # --------------------------------------------------

        move = board.move_from_csa(csa_move)

        if move == 0:
            raise ValueError(
                f"invalid CSA move: "
                f"{csa_move} "
                f"in {csa_path} "
                f"at ply={ply + 1}"
            )

        if not board.is_legal(move):
            raise ValueError(
                f"illegal CSA move: "
                f"{csa_move} "
                f"in {csa_path} "
                f"at ply={ply + 1}"
            )

        # 現在局面を保存
        sfen = board.sfen()

        # USI形式へ変換
        move_usi = cshogi.move_to_usi(move)

        ply += 1

        records.append(
            {
                "game_id": game_id,
                "ply": ply,
                "sfen": sfen,
                "move": move_usi,
                "result": result,
                "source": str(csa_path),
                "csa_move": csa_move,
                "color": color,
            }
        )

        board.push(move)

    return records


def convert_csa_directory(
    input_dir: str | Path,
    output_path: str | Path,
) -> int:

    csa_files = collect_csa_files(input_dir)

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_positions = 0
    total_games = 0
    skipped_games = 0

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for game_id, csa_path in enumerate(csa_files):

            try:
                records = parse_csa_file(
                    csa_path,
                    game_id,
                )

            except Exception as exc:

                skipped_games += 1

                print(
                    f"[SKIP] {csa_path}: {exc}"
                )

                continue

            for record in records:

                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            total_positions += len(records)
            total_games += 1

            if total_games % 25 == 0:

                print(
                    f"games={total_games} "
                    f"positions={total_positions} "
                    f"skipped={skipped_games}"
                )

    print()
    print("CSA変換完了")
    print("-" * 30)
    print(f"棋譜数     : {total_games}")
    print(f"スキップ数 : {skipped_games}")
    print(f"局面数     : {total_positions}")
    print(f"出力       : {output_path}")

    return total_positions


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Convert CSA game records "
            "to position JSONL."
        )
    )

    parser.add_argument(
        "--input",
        default="data/raw/wcsc",
    )

    parser.add_argument(
        "--output",
        default="data/processed/positions.jsonl",
    )

    args = parser.parse_args()

    convert_csa_directory(
        args.input,
        args.output,
    )


if __name__ == "__main__":
    main()