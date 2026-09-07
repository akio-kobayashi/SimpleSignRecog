"""手話動画の前処理を段階ごとにまとめたパッケージ。

学習データは次の順で処理します。

1. :mod:`landmark_extraction` で動画から126次元の座標を抽出する。
2. :mod:`missing_data` で検出できなかった座標を補間する。
3. :mod:`normalization` で位置・大きさ・向きをそろえる。
"""

# 各段階の関数は、その役割が分かるサブモジュールから明示的にimportします。
# 例: from src.preprocessing.missing_data import interpolate_missing_data
