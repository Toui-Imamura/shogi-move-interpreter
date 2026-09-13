"""
interpreter/dataset_io.py

学習用特徴量レコードのJSONL入出力。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


def write_jsonl(
    records: Iterable[Mapping[str, Any]],
    output_path: str | Path,
) -> int:
    """
    レコードをJSONL形式で保存する。

    Returns
    -------
    int
        保存したレコード数。
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(
                dict(record),
                file,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            file.write("\n")
            count += 1

    return count


def read_jsonl(
    input_path: str | Path,
) -> Iterator[dict[str, Any]]:
    """
    JSONLを1行ずつ読み込む。
    """

    path = Path(input_path)

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONLの{line_number}行目を解析できません: "
                    f"{path}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"JSONLの{line_number}行目は辞書である必要があります。"
                )

            yield record
