from __future__ import annotations

import sys
from pathlib import Path

# プロジェクトルートをPythonの検索パスへ追加
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cshogi

from interpreter.explainer import explain_move


def main() -> None:
    """
    初期局面の指し手を対象に説明候補を生成するデモ。
    """

    before = cshogi.Board()

    # USI文字列をそのまま渡す
    move_usi = "7g7f"

    result = explain_move(
        before=before,
        move_usi=move_usi,
    )

    print("=== 指し手説明デモ ===")
    print(f"指し手: {move_usi}")
    print()

    print("説明:")
    print(result.explanation)
    print()

    print("説明候補:")
    for index, candidate in enumerate(result.candidates, start=1):
        print(f"{index}. {candidate}")


if __name__ == "__main__":
    main()
