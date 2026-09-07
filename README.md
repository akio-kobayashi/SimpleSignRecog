# SimpleSignRecog

手話動画から両手のランドマークを抽出し、20クラスを学習・評価するプロジェクトです。

```text
MP4動画
  → MediaPipe Tasksで両手21点を抽出
  → 左右ラベルを動画単位で補正
  → 学習時に欠損補間・正規化・平滑化・特徴量生成
  → 交差検証で学習・評価
```

## 構成

実験コードとして管理する主要ディレクトリは次の4つです。

| 場所 | 内容 |
|---|---|
| `src/` | ランドマーク抽出、前処理、データセット、モデル |
| `scripts/` | サーバー上で前処理を一括実行するスクリプト |
| `notebooks/` | 前処理の説明用ノートブック |
| `tests/` | 前処理モジュールのテスト |

`data/`、`corrected/`、`experiments/`、`logs/`、`checkpoints/` などは生成物であり、Gitでは管理しません。Colabノートブックは `notebooks/preprocessing_colab.ipynb` に残していますが、現在の推奨手順はSSH先での `uv` 実行です。

## 1. サーバーでランドマークを抽出する

SSHで計算機へログインし、リポジトリを準備します。接続先、アカウント、秘密鍵はリポジトリへ保存しません。

```bash
ssh <account>@<server>
git clone https://github.com/akio-kobayashi/SimpleSignRecog.git
cd SimpleSignRecog
```

clone済みの場合は `git switch main` と `git pull --ff-only` で更新します。サーバーには `uv` が必要です。`uv --version` で利用できることを確認してください。

動画はサーバー上の次の場所に配置済みであることを前提とします。

```text
/srv/share/ego_sign_recog/videos/
├── subject_01/1～20/*.mp4
├── subject_02/1～20/*.mp4
└── subject_03/1～20/*.mp4
```

被験者ごとに実行します。

```bash
./scripts/run_preprocessing.sh subject_01
./scripts/run_preprocessing.sh subject_02
./scripts/run_preprocessing.sh subject_03
```

スクリプトは `uv` の隔離環境へPython 3.13、NumPy 2系、MediaPipe Tasksを準備し、抽出と左右ラベル補正を続けて実行します。依存関係は `requirements-preprocessing.txt` から読み込みます。元動画は変更しません。

```text
/srv/share/ego_sign_recog/processed/
├── data/<subject>/{metadata.csv, processed_data/1～20/*.npz}
└── corrected/<subject>/{metadata.csv, processed_data/1～20/*.npz}
```

`data` は抽出直後、`corrected` は左右ラベル補正後です。通常は `corrected` を学習に使います。検出条件は被験者IDの後ろへ渡せます。

```bash
./scripts/run_preprocessing.sh subject_01 --min-det 0.6 --min-trk 0.6
```

## 2. 抽出データと学習時前処理

各NPZの `landmarks` は形状 `(フレーム数, 126)` の `float32` 配列です。左手21点のXYZを列 `0:63`、右手21点を列 `63:126` に格納し、未検出座標は `NaN` とします。

`src/dataset.py` の `SignDataset` は、NPZを読み込むたびに次の処理を行います。

1. 座標ごとに時間方向へ線形補間する。
2. 学習データへ、設定に応じて左右反転、回転、ノイズを加える。
3. 手の位置・大きさ・向きを正規化する。
4. Savitzky–Golayフィルタで座標系列を平滑化する。
5. 位置、速度、加速度、指先間距離、正規化前の手首速度を結合する。標準設定は1フレーム392次元。
6. 残った `NaN` と無限値を0へ置換し、バッチ内の系列長を0埋めでそろえる。

処理は `src/preprocessing/` の抽出・欠損処理・正規化モジュールと `src/feature_engineering.py` に分かれています。正規化方式は `config.yaml` の `features.normalize_mode` で選びます。

| 設定値 | 処理 |
|---|---|
| `normalize_landmarks` | 手首を原点とし、手首と中指付け根の距離で尺度をそろえる |
| `canonical_normalize` | 手首を原点として尺度と掌の向きをそろえる |
| `current_wrist` | 各フレームの手首を原点とする論文方式 |
| `first_wrist` | 最初に検出した手首を共通原点とする論文方式 |

## 3. 学習・評価する

実験用YAMLで補正済みデータを指定します。

```yaml
data:
  metadata_path: /srv/share/ego_sign_recog/processed/corrected/subject_01/metadata.csv
  source_landmark_dir: /srv/share/ego_sign_recog/processed/corrected/subject_01/processed_data
```

```bash
python train.py --config <experiment.yaml> --cm-output-dir <results-directory>
```

`train.py` は各分割の学習・検証・テストを行い、チェックポイント、ログ、posteriogram、混同行列を保存します。混同行列は次のように集計します。

```bash
python aggregate_results.py <results-directory> --mode cv --config <experiment.yaml> --report-out <report.csv>
```

話者間評価には `train_cross_subject.py`、従来の機械学習モデルには `train_svm.py`、`train_rf.py`、`train_xgboost.py` を使います。動画、抽出済みNPZ、結果CSV、認証情報はコミットしません。

## 4. 前処理を検証する

前処理用依存関係を導入した環境では次を実行します。

```bash
uv run --isolated --python 3.13 --with-requirements requirements-preprocessing.txt python -m unittest tests.test_preprocessing
```
