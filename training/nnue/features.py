from __future__ import annotations

from dataclasses import dataclass

import cshogi


# cshogi 1.0.4 で確認済み
BLACK = cshogi.BLACK
WHITE = cshogi.WHITE


@dataclass(frozen=True)
class NNUEFeature:
    """
    NNUE用の疎な盤面特徴。

    king_square:
        視点側の玉のマス

    piece:
        cshogiの駒ID

    square:
        駒が存在するマス
    """

    king_square: int
    piece: int
    square: int


def extract_features(
    board: cshogi.Board,
    perspective: int,
) -> list[NNUEFeature]:
    """
    局面からNNUE用の特徴を抽出する。

    現段階では研究用の基本的な
    King + Piece + Square 表現を使用する。
    """

    king_square = board.king_square(perspective)

    features: list[NNUEFeature] = []

    for square in range(81):
        piece = board.piece(square)

        if piece == 0:
            continue

        features.append(
            NNUEFeature(
                king_square=int(king_square),
                piece=int(piece),
                square=int(square),
            )
        )

    return features


def extract_both_perspectives(
    board: cshogi.Board,
) -> tuple[list[NNUEFeature], list[NNUEFeature]]:
    """
    Black視点とWhite視点の特徴を取得する。
    """

    black_features = extract_features(
        board,
        BLACK,
    )

    white_features = extract_features(
        board,
        WHITE,
    )

    return black_features, white_features


KING_SQUARES = 81
PIECE_TYPES = 16
BOARD_SQUARES = 81


def feature_to_id(feature: NNUEFeature) -> int:
    """
    NNUEFeatureをEmbedding用の整数IDへ変換する。

    ID:
        king_square
        × piece
        × square
    """

    return (
        (
            feature.king_square * PIECE_TYPES
            + feature.piece
        )
        * BOARD_SQUARES
        + feature.square
    )


def num_feature_ids() -> int:
    """
    Embeddingに必要な特徴ID数。
    """

    return (
        KING_SQUARES
        * PIECE_TYPES
        * BOARD_SQUARES
    )
