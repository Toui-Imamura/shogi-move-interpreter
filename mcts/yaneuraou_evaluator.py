from __future__ import annotations

from training.teacher.yaneuraou import YaneuraOuEvaluator


class YaneuraOuMCTSEvaluator:
    """YaneuraOuをMCTSの局面評価器として利用するアダプタ。"""

    def __init__(
        self,
        *,
        engine_path: str,
        eval_dir: str | None = None,
        depth: int = 8,
        threads: int = 1,
        hash_mb: int = 256,
        timeout: int = 60,
    ) -> None:
        self.engine = YaneuraOuEvaluator(
            engine_path=engine_path,
            eval_dir=eval_dir,
            depth=depth,
            threads=threads,
            hash_mb=hash_mb,
            timeout=timeout,
        )

    def __call__(self, board) -> float:
        """cshogi.BoardをYaneuraOuのBlack視点評価値へ変換する。"""
        result = self.engine.evaluate_sfen(
            board.sfen()
        )

        return float(
            result["score_cp_black"]
        )

    def close(self) -> None:
        self.engine.close()

    def __enter__(self) -> "YaneuraOuMCTSEvaluator":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()
