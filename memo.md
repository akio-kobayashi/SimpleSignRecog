apptainerイメージをSIF（配布済み）とする

1. ApptainerコンテナをGPU付きで起動
apptainer exec --nv ${SIF} bash

2. Python環境を構築（初回のみ）
bash ./bin/init.sh

3. 仮想環境を作成（初回のみ）
bash ./bin/create_venv.sh

4. 仮想環境を有効化して作業
source ~/.venv/bin/activate

5. 任意のリポジトリをcloneして作業（初回のみ）
git clone git clone https://github.com/akio-kobayashi/SimpleSignRecog.git
cd ./SimpleSignRecog/

