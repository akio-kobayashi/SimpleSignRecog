# 必要なライブラリをインポートします
from argparse import ArgumentParser  # コマンドライン引数を解析するために使用します
import pandas as pd  # データ操作と分析のためにpandasを使用します

# --- コマンドライン引数の設定 ---
# ArgumentParserオブジェクトを作成し、スクリプトの説明を記述します
parser = ArgumentParser(description="メタデータCSVファイルから、クラスごとに指定された数のサンプルを抽出してサブセットを作成します。")

# --max_num: 各クラスから抽出する最大のサンプル数を指定します
parser.add_argument("--max_num", type=int, default=20, help="1クラスあたりの最大サンプル数")

# --output: 出力するCSVファイルの名前を指定します
parser.add_argument("--output", type=str, default="metadata_subset.csv", help="出力CSVファイル名")

# --input: 入力となる元のメタデータCSVファイルを指定します (必須)
parser.add_argument("--input", type=str, required=True, help="入力CSVファイル名")

# コマンドラインから渡された引数を解析します
args = parser.parse_args()

# --- メタデータの読み込み ---
# pandasのread_csv関数を使って、指定された入力CSVファイルを読み込み、データフレームに格納します
print(f"入力ファイル: {args.input} を読み込んでいます...")
df = pd.read_csv(args.input)

# --- データフレームのサブセット作成 ---
# groupby("class_label")で、データをクラスラベルごとにグループ化します
# .apply() を使って、各グループに対してサンプリング処理を適用します
# lambda x: x.sample(...) は、各グループ(x)からランダムにサンプルを抽出する無名関数です
#   - n=min(len(x), args.max_num) は、グループのサンプル数と指定された最大数のうち、小さい方を抽出数とします
#     （これにより、サンプル数がmax_numより少ないクラスでもエラーなく動作します）
#   - random_state=42 は、再現性を確保するために乱数シードを固定します
print(f"各クラスから最大 {args.max_num} 個のサンプルをランダムに抽出しています...")
subset_df = (
    df.groupby("class_label", group_keys=False)
      .apply(lambda x: x.sample(n=min(len(x), args.max_num), random_state=42))
)

# --- サブセットの保存 ---
# 作成したサブセットのデータフレームを、指定された出力ファイル名でCSVとして保存します
# index=Falseは、データフレームのインデックスをCSVファイルに書き込まないようにする設定です
subset_df.to_csv(args.output, index=False)

# --- 完了メッセージ ---
print(f"メタデータのサブセットを作成し、{args.output} に保存しました。")
print(f"作成されたファイルの行数: {len(subset_df)}")
