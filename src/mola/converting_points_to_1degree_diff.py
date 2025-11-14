import numpy as np
import pandas as pd

df_csv = pd.read_csv("mars_landing_sites_topo.csv")


# --- 1° ---
df_csv['lon'] = df_csv['lon'].round(0)
df_csv['lat'] = df_csv['lat'].round(0)


# --- 1° x 1° ---
agg_df = df_csv.groupby(['lon', 'lat']).agg({
'rank': 'min' # jeśli choć jeden punkt jest idealny (1), kwadrat traktujemy jako najlepszy
}).reset_index()


agg_df.to_csv("mars_landing_sites_1deg.csv", index=False)
print("Dane zgrupowane do siatki 1° zapisane do mars_landing_sites_1deg.csv")