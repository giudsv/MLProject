import pandas as pd

# Caricare il dataset
df = pd.read_csv("../dataset/finalDataset.csv")

# Dizionario per conversione delle unità di misura e rinomina colonne
conversion_map = {
    "RedHeightCms": ("RedHeightMt", 0.01),  # cm -> m
    "BlueHeightCms": ("BlueHeightMt", 0.01),  # cm -> m
    "RedReachCms": ("RedReachMt", 0.01),  # cm -> m
    "BlueReachCms": ("BlueReachMt", 0.01),  # cm -> m
    "RedWeightLbs": ("RedWeightKg", 0.453592),  # lbs -> kg
    "BlueWeightLbs": ("BlueWeightKg", 0.453592)  # lbs -> kg
}

# Applicare le conversioni e rinominare le colonne mantenendo l'ordine originale
for old_col, (new_col, factor) in conversion_map.items():
    df.insert(df.columns.get_loc(old_col) + 1, new_col, (df[old_col] * factor).round(2))  # Convertire e arrotondare
    df.drop(columns=[old_col], inplace=True)  # Eliminare la colonna originale

# Salvare il nuovo dataset
scaled_file_path = "../dataset/ScaledDataset.csv"
df.to_csv(scaled_file_path, index=False)

print(f"Dataset normalizzato salvato in: {scaled_file_path}")
