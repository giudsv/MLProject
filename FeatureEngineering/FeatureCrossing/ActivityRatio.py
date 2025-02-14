import pandas as pd

# Caricare il dataset
df = pd.read_csv('../../dataset/finalDataset.csv')

# Calcolare il numero totale di match per Red e Blue fighter
df["RedTotalFights"] = df["RedWins"] + df["RedLosses"] + df["RedDraws"]
df["BlueTotalFights"] = df["BlueWins"] + df["BlueLosses"] + df["BlueDraws"]

# Evitare divisioni per zero
df["ActivityRatio_Red"] = df["RedTotalFights"] / (df["DaysSinceFirstFight"] + 1)
df["ActivityRatio_Blue"] = df["BlueTotalFights"] / (df["DaysSinceFirstFight"] + 1)

# Differenza tra i due fighter
df["ActivityRatioDif"] = df["ActivityRatio_Red"] - df["ActivityRatio_Blue"]

# Salvare il dataset aggiornato
df.to_csv('../../dataset/finalDataset.csv', index=False)
