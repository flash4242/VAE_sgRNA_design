#!/usr/bin/env python3
import pandas as pd

file_path = 'data_all_ont.xlsx'

# Load all sheets into a dictionary of DataFrames
sheets = pd.read_excel(file_path, sheet_name=None)

# Combine all sheets into a single DataFrame
combined_df = pd.concat(sheets.values(), ignore_index=True)

# Retain only the "sgRNA" and "Normalized efficacy" columns
result_df = combined_df[['sgRNA', 'Normalized efficacy']]

# Display or save the result
print(result_df)

result_df.to_csv('data_for_ont_training.csv', index=False)
print("Data has been saved to 'data_for_ont_training.csv'")
