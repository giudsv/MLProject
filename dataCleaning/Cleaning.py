import pandas as pd

# Caricare il dataset
df = pd.read_csv("../dataset/ufc-master.csv")

# Lista delle colonne da rimuovere
cols_to_remove = {
    "RedOdds", "BlueOdds", "RedExpectedValue", "BlueExpectedValue",
    "RedDecOdds", "BlueDecOdds", "RSubOdds", "BSubOdds", "RKOOdds", "BKOOdds"
}


df = df.drop(columns=cols_to_remove, errors="ignore")


output_path = "../dataset/finalDataset.csv"

# Salvare il dataset in un nuovo file CSV
df.to_csv(output_path, index=False)

