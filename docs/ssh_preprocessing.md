# SSHで前処理を実行する

学生はSSHで計算機へログインし、サーバー上のDockerコンテナで特徴量を抽出する。接続先、アカウント名、秘密鍵はリポジトリへ保存しない。

## 1. サーバーへログインする

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

## 3. 入出力ディレクトリ

動画はサーバー上の次の場所にある。

```text
/srv/share/ego_sign_recog/videos/
├── subject_01/1～20/*.mp4
├── subject_02/1～20/*.mp4
└── subject_03/1～20/*.mp4
```

生成物は `/srv/share/ego_sign_recog/processed` 以下へ保存する。元動画は変更しない。

## 4. 前処理を実行する

担当する被験者IDを1つ指定する。

```bash
./scripts/run_preprocessing.sh subject_01
```

`subject_02`、`subject_03`も同じ形式で実行する。スクリプトは次を順に行う。

1. Python 3.13、NumPy 2系、MediaPipe Tasksを含むDockerイメージをbuildする。
2. `/srv/share/ego_sign_recog/videos/<被験者ID>` の全MP4から左右21点の画像座標を抽出する。
3. 抽出直後のNPZと `metadata.csv` を `processed/data/<被験者ID>` に保存する。
4. 左右ラベルを補完した動画を判定し、補正済み一式を `processed/corrected/<被験者ID>` に保存する。

```text
/srv/share/ego_sign_recog/processed/
├── data/
│   ├── subject_01/{metadata.csv, processed_data/1～20/*.npz}
│   ├── subject_02/{metadata.csv, processed_data/1～20/*.npz}
│   └── subject_03/{metadata.csv, processed_data/1～20/*.npz}
└── corrected/
    ├── subject_01/{metadata.csv, processed_data/1～20/*.npz}
    ├── subject_02/{metadata.csv, processed_data/1～20/*.npz}
    └── subject_03/{metadata.csv, processed_data/1～20/*.npz}
```

NPZの `landmarks` は `(フレーム数, 126)` のfloat32配列である。左手21点を列 `0:63`、右手21点を列 `63:126` に格納し、未検出座標は `NaN` とする。欠損補間、正規化、特徴量結合は学習時に `SignDataset` が行う。

検出条件を変える場合は末尾へオプションを渡す。

```bash
./scripts/run_preprocessing.sh subject_01 --min-det 0.6 --min-trk 0.6
```

## 5. 終了する

```bash
exit
```
