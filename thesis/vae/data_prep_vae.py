#!/usr/bin/env python3

import pandas as pd

df = pd.read_csv("hl60/data_for_vae_hl60.csv")
df = df[["sgRNA"]]
df.to_csv("hl60/data_for_vae_hl60.csv", index=False)