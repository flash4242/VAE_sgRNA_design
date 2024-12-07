#!/usr/bin/env python3


import pandas as pd

# Read each CSV file
csv1 = pd.read_csv("ont_hek293t/data_for_ont_hek293t.csv")
csv2 = pd.read_csv("ont_hela/data_for_ont_hela.csv")
csv3 = pd.read_csv("ont_hl60/data_for_ont_hl60.csv")

# Concatenate them
unified_csv = pd.concat([csv1, csv2, csv3])

# Save to a new CSV file
unified_csv.to_csv("all_except_hct116/all_data_except_hct116_ont.csv", index=False)
