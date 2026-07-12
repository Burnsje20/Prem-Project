import json
import pandas as pd

with open("Premier_League_Seasons_Data.json", "r") as f:
    raw_data = json.load(f)
    all_matches = []
    for year, matches in raw_data.items():
        all_matches.extend(matches)
df = pd.DataFrame(all_matches)
print(df.head())