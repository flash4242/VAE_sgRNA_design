#!/usr/bin/env python3


import pandas as pd

# Read each CSV file
csv1 = pd.read_csv("hek293t/data_for_vae_hek293t.csv")
csv2 = pd.read_csv("hela/data_for_vae_hela.csv")
csv3 = pd.read_csv("hl60/data_for_vae_hl60.csv")

# Concatenate them
unified_csv = pd.concat([csv1, csv2, csv3])

# Save to a new CSV file
unified_csv.to_csv("all_data_except_hct116_vae.csv", index=False)
