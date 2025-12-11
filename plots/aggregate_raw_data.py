from os import listdir
from os.path import join

import pandas as pd
pd.set_option("display.float_format", lambda x: f"{x:.16f}")

RAW_DATA: str = join("plots", "project", "raw_data")
AGGREGATED: str = join("plots", "project", "aggregated_data", "aggregated.csv")

aggregated_df = []

for item in listdir(RAW_DATA):
    
    if not item.endswith(".csv"):
        continue

    item = join(RAW_DATA, item)

    df = pd.read_csv(item)
    
    columns = df.columns.tolist()
    columns = [column for column in columns if not any(key in column for key in ["__MIN", "__MAX", "Step"])]

    df = df[columns]
    
    aggregated_df.append(df)

aggregated_df = pd.concat(aggregated_df, axis=1)
aggregated_df.columns = [column.replace("[GPU] Wide Deep Network - ", "")for column in aggregated_df.columns]

aggregated_df.to_csv(AGGREGATED, index=False)