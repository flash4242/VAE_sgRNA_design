#!/usr/bin/env python3
import pandas as pd

def xlsx_to_csv(xlsx_file, csv_file):
    # Read the Excel file
    df = pd.read_excel(xlsx_file, engine='openpyxl')
    
    # Save the DataFrame to a CSV file
    df.to_csv(csv_file, index=False)

# Example usage:
xlsx_file = 'merged_offt.xlsx'  # Replace with your .xlsx file
csv_file = 'merged_offt.csv'   # Replace with the desired .csv output file
xlsx_to_csv(xlsx_file, csv_file)

print(f"File converted successfully from {xlsx_file} to {csv_file}.")