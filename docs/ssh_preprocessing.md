# SSHで前処理を実行する

Colabは使用せず、学生がSSHで計算機へログインし、サーバー上のDockerコンテナで特徴量を抽出する。接続先、アカウント名、秘密鍵、元動画はリポジトリへ保存しない。

## 1. サーバーへログインする

ローカルPCの端末から、割り当てられた接続先とアカウントを指定する。

```bash
ssh <account>@<server>
```

秘密鍵を明示する場合は `ssh -i <private-key> <account>@<server>` とする。秘密鍵はサーバーやGitリポジトリへコピーしない。

## 2. コードを準備する

初回だけリポジトリをcloneする。clone済みの場合はmainを更新する。

```bash
git clone https://github.com/akio-kobayashi/SimpleSignRecog.git
cd SimpleSignRecog
git switch main
git pull --ff-only
```

## 3. 動画を配置する

入力ディレクトリ直下の `1`～`20` に各クラスのMP4を置く。動画はGitリポジトリの外に置く。

```text
/path/to/subject_videos/
├── 1/*.mp4
├── 2/*.mp4
└── .../20/*.mp4
```

## 4. 前処理を実行する

元動画、出力ルート、匿名の被験者IDを指定する。

```bash
./scripts/run_preprocessing.sh \
  /path/to/subject_videos \
  /path/to/SimpleSignRecogData \
  subject_03
```

スクリプトは次を順に実行する。

1. Python 3.13、NumPy 2系、MediaPipe Tasksを含むDockerイメージをbuildする。
2. 公式Hand Landmarkerモデルで全MP4から左右21点の画像座標を抽出する。
3. 抽出直後のNPZと `metadata.csv` を `data/<被験者ID>` に保存する。
4. 左右ラベルの補完が発生した動画を判定し、補正済み一式を `corrected/<被験者ID>` に保存する。

```text
/path/to/SimpleSignRecogData/
├── data/subject_03/
│   ├── metadata.csv
│   └── processed_data/1～20/*.npz
└── corrected/subject_03/
    ├── metadata.csv
    └── processed_data/1～20/*.npz
```

NPZの `landmarks` は `(フレーム数, 126)` のfloat32配列である。左手21点を列 `0:63`、右手21点を列 `63:126` に格納し、未検出座標は `NaN` とする。欠損補間、正規化、特徴量結合は学習時に `SignDataset` が行う。

MediaPipeの検出条件を変える場合は末尾へオプションを渡す。

```bash
./scripts/run_preprocessing.sh \
  /path/to/subject_videos \
  /path/to/SimpleSignRecogData \
  subject_03 \
  --min-det 0.6 --min-trk 0.6
```

## 5. 終了する

```bash
exit
```
