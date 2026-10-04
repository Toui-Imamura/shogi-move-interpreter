import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.teacher.yaneuraou import YaneuraOuEvaluator


ENGINE_PATH = "/workspace/YaneuraOu/source/YaneuraOu-by-gcc"
EVAL_DIR = "/workspace/YaneuraOu/source/eval"


def main():
    evaluator = YaneuraOuEvaluator(
        engine_path=ENGINE_PATH,
        eval_dir=EVAL_DIR,
        depth=8,
        threads=1,
        hash_mb=256,
        timeout=60,
    )

    try:
        import cshogi

        board = cshogi.Board()

        result = evaluator.evaluate_sfen(board.sfen())

        print("=== YaneuraOu test ===")
        print(f"score_cp_side_to_move: {result['score_cp_side_to_move']}")
        print(f"score_cp_black: {result['score_cp_black']}")
        print(f"score_raw: {result['score_raw']}")
        print(f"mate: {result['mate']}")
        print(f"depth: {result['depth']}")
        print(f"seldepth: {result['seldepth']}")
        print(f"bestmove: {result['bestmove']}")
        print("PASS")

    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
