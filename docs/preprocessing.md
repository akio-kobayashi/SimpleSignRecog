# 学習データの前処理

前回の実験と同じ前処理を、役割ごとのモジュールに分けています。

```text
MP4動画
  ↓ landmark_extraction.py
左右21点 × XYZ = 126次元のランドマーク（未検出はNaN）
  ↓ missing_data.py
時間方向の線形補間
  ↓ normalization.py
位置・大きさ・向きの正規化
  ↓ feature_engineering.py
平滑化と392次元の特徴量生成
```

## 1. MediaPipeによるランドマーク抽出

実装は `src/preprocessing/landmark_extraction.py` にあります。左手を0～62、
右手を63～125に格納します。MediaPipeが左右を返さない場合の推測規則、
信頼度の既定値（検出0.5、追跡0.5）、モデル複雑度1は前回と同じです。

データセット全体を変換するコマンドは従来どおりです。

```bash
python src/process_videos.py \
  --input_root_dir /path/to/subject_videos \
  --output_base_dir ./data/subject
```

出力は動画ごとの `processed_data/<クラス>/<動画名>.npz` と
`metadata.csv` です。

## 2. 欠損処理

`src/preprocessing/missing_data.py` の `interpolate_missing_data` が、各座標を
フレーム方向に線形補間します。先頭と末尾のNaNは最寄りの検出値で埋めます。
動画を通して一度も検出されない座標のNaNは残り、学習パイプラインの最後で
0に置換されます。

## 3. 正規化

`src/preprocessing/normalization.py` に4方式があります。前回の実験設定は
`canonical_normalize` です。

| 設定値 | 関数 | 処理 |
|---|---|---|
| `normalize_landmarks` | `normalize_landmarks` | 手首を原点、0番–9番の距離を1にする |
| `canonical_normalize` | `canonical_normalize_landmarks` | 手首を原点、0番–17番で尺度を統一し、掌の向きもそろえる |
| `current_wrist` | `normalize_by_current_wrist` | フレームごとの手首を原点にする |
| `first_wrist` | `normalize_by_first_wrist` | 最初に検出した手首を全フレーム共通の原点にする |

実際の学習時の呼び出し順は `src/dataset.py` で確認できます。データ拡張は
欠損補間の後、正規化の前に適用されます。
