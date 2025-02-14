import pandas as pd

# Caricare il dataset
df = pd.read_csv('../../dataset/finalDataset.csv')

# Assicurarsi che i valori di DaysSinceLastFight_Red e DaysSinceLastFight_Blue siano numerici
df["DaysSinceLastFight_Red"] = pd.to_numeric(df["DaysSinceLastFight_Red"], errors='coerce')
df["DaysSinceLastFight_Blue"] = pd.to_numeric(df["DaysSinceLastFight_Blue"], errors='coerce')

# Creare la colonna "RestTimeDif" e calcolare la differenza
df["RestTimeDif"] = df["DaysSinceLastFight_Red"] - df["DaysSinceLastFight_Blue"]

# Salvare il dataset aggiornato con la nuova feature
df.to_csv('../../dataset/finalDataset.csv', index=False)

print("✅ Feature 'RestTimeDif' aggiunta con successo e dataset salvato!")
