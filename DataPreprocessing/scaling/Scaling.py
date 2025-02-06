import pandas as pd

# Caricare il dataset
df = pd.read_csv("../../dataset/finalDataset.csv")

# Convertire la colonna Date in formato datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Gestisce eventuali errori di conversione

# Calcolare la data minima nel dataset
min_date = df['Date'].min()

# Funzione per convertire i tempi nel formato 'mm:ss' in secondi
def convert_time_to_seconds(time_str):
    try:
        minutes, seconds = map(int, time_str.split(":"))
        return minutes * 60 + seconds
    except Exception as e:
        return 0  # In caso di errore, restituisci 0

# Dizionario per conversione delle unità di misura e rinomina colonne
conversion_map = {
    "RedHeightCms": ("RedHeightMt", 0.01),  # cm -> m
    "BlueHeightCms": ("BlueHeightMt", 0.01),  # cm -> m
    "RedReachCms": ("RedReachMt", 0.01),  # cm -> m
    "BlueReachCms": ("BlueReachMt", 0.01),  # cm -> m
    "RedWeightLbs": ("RedWeightKg", 0.453592),  # lbs -> kg
    "BlueWeightLbs": ("BlueWeightKg", 0.453592),  # lbs -> kg
    "TotalFightTimeSecs": ("TotalFightTimeMins", 1 / 60),  # sec -> min
    "FinishRoundTime": ("FinishRoundInSeconds", lambda x: convert_time_to_seconds(x)),  # mm:ss -> secondi
    "Date": ("DaysSinceFirstFight", lambda x: (x - min_date).days)  # Trasformare la data in giorni dalla minima
}

# Applicare le conversioni e rinominare le colonne mantenendo l'ordine originale
for old_col, (new_col, factor) in conversion_map.items():
    if callable(factor):  # Se il valore è una funzione (per la data o tempo)
        df.insert(df.columns.get_loc(old_col) + 1, new_col, df[old_col].apply(factor))
    else:  # Per tutte le altre conversioni numeriche
        df.insert(df.columns.get_loc(old_col) + 1, new_col, (df[old_col] * factor).round(2))

    df.drop(columns=[old_col], inplace=True)  # Eliminare la colonna originale

# Salvare il nuovo dataset
scaled_file_path = "../../dataset/finalDataset.csv"
df.to_csv(scaled_file_path, index=False)

print(f"Dataset con data trasformata, unità di misura convertite e tempo in secondi salvato in: {scaled_file_path}")
