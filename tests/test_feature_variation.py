import cshogi

from features.transition import make_usi_variation
from features.attack import (
    f13_material_gain_process,
    f15_attack_continuity,
)


def test_f13_f15_variation():
    board = cshogi.Board()

    variation = make_usi_variation(
        board,
        [
            "7g7f",
            "3c3d",
            "2g2f",
        ],
    )

    f13 = f13_material_gain_process(variation)
    f15 = f15_attack_continuity(variation)

    print("F13:", f13)
    print("F15:", f15)

    assert f13 is not None
    assert f15 is not None
