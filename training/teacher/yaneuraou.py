from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional


class YaneuraOuError(RuntimeError):
    """YaneuraOuとの通信に関するエラー。"""


class YaneuraOuEvaluator:
    """
    YaneuraOuをUSIプロトコルで操作し、
    指定したSFEN局面の評価値を取得する。

    評価値は「黒から見たcp値」に正規化して返す。
    """

    def __init__(
        self,
        engine_path: str | Path,
        eval_dir: str | Path | None = None,
        depth: int = 8,
        threads: int = 1,
        hash_mb: int = 256,
        timeout: float = 60.0,
    ) -> None:
        self.engine_path = Path(engine_path)
        self.eval_dir = Path(eval_dir) if eval_dir is not None else None
        self.depth = depth
        self.threads = threads
        self.hash_mb = hash_mb
        self.timeout = timeout

        if not self.engine_path.exists():
            raise FileNotFoundError(
                f"YaneuraOuが見つかりません: {self.engine_path}"
            )

        self.process: Optional[subprocess.Popen[str]] = None

    def start(self) -> None:
        """YaneuraOuを起動してUSI初期化を行う。"""

        if self.process is not None:
            return

        self.process = subprocess.Popen(
            [str(self.engine_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        self._send("usi")

        usiok = self._read_until("usiok")
        if usiok is None:
            raise YaneuraOuError("YaneuraOuからusiokを取得できませんでした。")

        # 定跡を使わず、評価値＋探索だけで教師値を生成する。
        self._send("setoption name USI_OwnBook value false")

        self._send(f"setoption name Threads value {self.threads}")
        self._send(f"setoption name USI_Hash value {self.hash_mb}")

        if self.eval_dir is not None:
            self._send(
                f"setoption name EvalDir value {self.eval_dir}"
            )

        self._send("isready")

        readyok = self._read_until("readyok")
        if readyok is None:
            raise YaneuraOuError(
                "YaneuraOuからreadyokを取得できませんでした。"
            )

    def evaluate_sfen(self, sfen: str) -> dict:
        """
        SFEN局面を評価する。

        Returns:
            {
                "score_cp_side_to_move": int,
                "score_cp_black": int,
                "score_raw": str,
                "mate": int | None,
                "depth": int | None,
                "seldepth": int | None,
                "bestmove": str | None,
            }
        """

        if self.process is None:
            self.start()

        # SFENから手番を取得。
        # SFENの4項目目が b / w。
        parts = sfen.split()

        if len(parts) < 2:
            raise ValueError(f"不正なSFENです: {sfen}")

        side_to_move = parts[1]

        self._send(f"position sfen {sfen}")
        self._send(f"go depth {self.depth}")

        best_info = None
        bestmove = None

        while True:
            line = self._readline()

            if line is None:
                raise YaneuraOuError(
                    "YaneuraOuが終了しました。"
                )

            line = line.strip()

            if line.startswith("info ") and "score " in line:
                parsed = self._parse_info_line(line)

                if parsed is not None:
                    best_info = parsed

            if line.startswith("bestmove "):
                tokens = line.split()

                if len(tokens) >= 2:
                    bestmove = tokens[1]

                break

        if best_info is None:
            raise YaneuraOuError(
                f"評価値を取得できませんでした。\nSFEN: {sfen}"
            )

        score_side = best_info["score_cp"]

        # YaneuraOuのscore cpは基本的に手番側から見た値。
        # 黒番ならそのまま、白番なら符号を反転して
        # 「黒から見た評価値」に統一する。
        if side_to_move == "b":
            score_black = score_side
        elif side_to_move == "w":
            score_black = -score_side
        else:
            raise ValueError(
                f"SFENの手番が不正です: {side_to_move}"
            )

        return {
            "score_cp_side_to_move": score_side,
            "score_cp_black": score_black,
            "score_raw": best_info["score_raw"],
            "mate": best_info["mate"],
            "depth": best_info["depth"],
            "seldepth": best_info["seldepth"],
            "bestmove": bestmove,
        }

    def close(self) -> None:
        """YaneuraOuを終了する。"""

        if self.process is None:
            return

        try:
            self._send("quit")
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()

        self.process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _send(self, command: str) -> None:
        if self.process is None or self.process.stdin is None:
            raise YaneuraOuError("YaneuraOuが起動していません。")

        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _readline(self) -> Optional[str]:
        if self.process is None or self.process.stdout is None:
            return None

        return self.process.stdout.readline()

    def _read_until(self, target: str) -> Optional[str]:
        while True:
            line = self._readline()

            if line is None:
                return None

            line = line.strip()

            if line == target:
                return line

    @staticmethod
    def _parse_info_line(line: str) -> Optional[dict]:
        """
        例:
        info depth 8 seldepth 15 multipv 1 score cp 100 ...
        """

        score_match = re.search(
            r"\bscore\s+(cp|mate)\s+(-?\d+)",
            line,
        )

        if score_match is None:
            return None

        score_type = score_match.group(1)
        score_value = int(score_match.group(2))

        depth_match = re.search(r"\bdepth\s+(\d+)", line)
        seldepth_match = re.search(r"\bseldepth\s+(\d+)", line)

        depth = (
            int(depth_match.group(1))
            if depth_match
            else None
        )

        seldepth = (
            int(seldepth_match.group(1))
            if seldepth_match
            else None
        )

        if score_type == "cp":
            score_cp = score_value
            mate = None
            score_raw = f"cp {score_value}"
        else:
            # mateの場合はcpとして扱わず、別途記録する。
            mate = score_value

            # 勝ち負けの方向を保つため、
            # 十分大きなcp値へ暫定変換する。
            if score_value > 0:
                score_cp = 100000
            else:
                score_cp = -100000

            score_raw = f"mate {score_value}"

        return {
            "score_cp": score_cp,
            "score_raw": score_raw,
            "mate": mate,
            "depth": depth,
            "seldepth": seldepth,
        }