import pandas as pd

# Caricare il dataset
df = pd.read_csv("../../dataset/finalDataset.csv")

# Selezioniamo le colonne categoriche per One-Hot Encoding
categorical_columns = ['BlueStance','RedStance', 'RedFighter', 'BlueFighter', 'Location', 'Country', 'Winner', 'TitleBout', 'WeightClass', 'Gender', 'Finish']

# Applicare One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Specifica il percorso di output del nuovo file CSV
output_path = "../../dataset/finalDataset.csv"

# Salvare il dataset trasformato in un nuovo file CSV
df_encoded.to_csv(output_path, index=False)

print("One-Hot Encoding completato e file salvato come finalDataset_encoded.csv")
