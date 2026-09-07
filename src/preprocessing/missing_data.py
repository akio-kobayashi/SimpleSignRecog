"""MediaPipeで検出できなかったランドマークの補間。"""

import numpy as np
import pandas as pd


def interpolate_missing_data(landmarks: np.ndarray) -> np.ndarray:
    """時間方向の線形補間でNaNを埋める。

    動画の先頭・末尾にあるNaNは、最も近い検出値で埋めます。ある座標が
    動画全体で一度も検出されなかった場合、その列のNaNは残ります。これは
    前回の実験と同じ挙動です。

    Args:
        landmarks: 形状 ``(フレーム数, 126)`` のランドマーク。

    Returns:
        入力と同じ形状の補間済み配列。
    """
    dataframe = pd.DataFrame(landmarks)
    return dataframe.interpolate(
        method="linear", limit_direction="both", axis=0
    ).to_numpy()
