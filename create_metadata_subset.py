import pandas as pd

# Load the metadata
df = pd.read_csv("metadata.csv")

# Group by class_label and take the first 20 samples from each group
subset_df = df.groupby("class_label").head(20)

# Save the subset to a new CSV file
subset_df.to_csv("metadata_subset.csv", index=False)

print("Subset of metadata has been created and saved to metadata_subset.csv")
