from training.teacher.yaneuraou import YaneuraOuEvaluator


ENGINE_PATH = (
    "/workspace/YaneuraOu/source/YaneuraOu-by-gcc"
)

EVAL_DIR = (
    "/workspace/YaneuraOu/source/eval"
)


SFEN = (
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/"
    "PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
)


def main():
    with YaneuraOuEvaluator(
        engine_path=ENGINE_PATH,
        eval_dir=EVAL_DIR,
        depth=8,
        threads=1,
        hash_mb=256,
    ) as evaluator:

        result = evaluator.evaluate_sfen(SFEN)

        print("評価結果")
        print("--------------------")

        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()