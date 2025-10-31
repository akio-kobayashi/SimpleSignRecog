from argparse import ArgumentParser
import pandas as pd

parser = ArgumentParser()
parser.add_argument("--max_num", type=int, default=20, help="１クラスあたりの最大サンプル数")
parser.add_argument("--output", type=str, default="metadata_subset.csv", help="出力CSV")
args = parser.parse_args()

# Load the metadata
df = pd.read_csv("metadata.csv")

# Group by class_label and take the first max_num samples from each group
#subset_df = df.groupby("class_label").head(args.max_num)
subset_df = (
    df.groupby("class_label", group_keys=False)
      .apply(lambda x: x.sample(n=min(len(x), args.max_num), random_state=42))
)

# Save the subset to a new CSV file
subset_df.to_csv(args.output, index=False)

print("Subset of metadata has been created and saved to metadata_subset.csv")
