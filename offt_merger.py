#!/usr/bin/env python3

import pandas as pd

# Load the two Excel files
file1 = 'off_targget1.xlsx'  # First file with "Target ID", "OT", and "Cleavage Frequency"
file2 = 'off_targget2.xlsx'  # Second file with "Target ID" and "Target sgRNA"

# Read both Excel sheets
df1 = pd.read_excel(file1)  # First file
df2 = pd.read_excel(file2)  # Second file

# Merge the two dataframes based on "Target ID"
merged_df = pd.merge(df1, df2, on='Target ID', how='left')

# Display the result
print(merged_df)

# Optionally, you can save the merged result to a new Excel file
merged_df.to_excel('merged_offt.xlsx', index=False)
