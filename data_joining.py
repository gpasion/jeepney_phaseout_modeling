import pandas as pd
import glob
import os

# Set the path where your CSV files are stored
csv_folder = '/Users/gpasion/Documents/INST414/dataset_uncombined'  # update this
csv_files = glob.glob(os.path.join(csv_folder, '*.csv'))

# Read and combine all CSV files
df_list = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)

# Save to a new CSV if needed
combined_df.to_csv('/Users/gpasion/Documents/INST414/combined_output.csv', index=False)

