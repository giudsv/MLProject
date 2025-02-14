import pandas as pd

# Caricare il dataset
df = pd.read_csv("../../dataset/finalDataset.csv")

# Sostituire i valori mancanti con 0 per ogni colonna
df.fillna("0", inplace=True)

# Verifica che i valori mancanti siano stati sostituiti
output_path = "../../dataset/finalDataset.csv"

# Salvare il dataset in un nuovo file CSV
df.to_csv(output_path, index=False)