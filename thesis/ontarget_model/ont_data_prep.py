#!/usr/bin/env python3
import pandas as pd

file_path = 'data_all_ont.xlsx'

# Load all sheets into a dictionary of DataFrames
# For loading different sheets, use the different values for sheet_name attribute: hct116, hek293t, hela, hl60 or for loading all sheets: None
sheet = pd.read_excel(file_path, sheet_name="hl60")

# Combine all sheets into a single DataFrame
# combined_df = pd.concat(sheets.values(), ignore_index=True) # ------ USE ONLY IF ALL SHEETS ARE LOADED

# Retain only the "sgRNA" and "Normalized efficacy" columns
result_df = sheet[['sgRNA', 'Normalized efficacy']]

# Display or save the result
print(result_df)

result_df.to_csv('ont_hl60/data_for_ont_hl60.csv', index=False)
print("Data has been saved to 'ont_hl60/data_for_ont_hl60.csv'")
