import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Caricare il dataset
df = pd.read_csv("../../dataset/finalDataset.csv")

# Creare un oggetto LabelEncoder
label_encoder = LabelEncoder()

# Concatenare le due colonne per ottenere un mapping coerente tra RedFighter e BlueFighter
all_fighters = pd.concat([df['RedFighter'], df['BlueFighter']], axis=0).unique()

# Fit del LabelEncoder su tutti i nomi degli atleti
label_encoder.fit(all_fighters)

# Creare un DataFrame con il mapping atleta -> label
fighters_mapping = pd.DataFrame({'Fighter': all_fighters, 'Label': label_encoder.transform(all_fighters)})

# Salvare il mapping in un file CSV
fighters_mapping.to_csv("../../dataset/fighters_labels.csv", index=False)


# Applicare lo stesso encoding a entrambe le colonne
df['RedFighter'] = label_encoder.transform(df['RedFighter'])
df['BlueFighter'] = label_encoder.transform(df['BlueFighter'])

# Applicare Label Encoding anche a Location e Country
df['Location'] = label_encoder.fit_transform(df['Location'])
df['Country'] = label_encoder.fit_transform(df['Country'])

# Selezioniamo le colonne categoriche per One-Hot Encoding
categorical_columns = ['BlueStance', 'RedStance', 'Winner', 'TitleBout', 'WeightClass', 'Gender', 'Finish', 'BetterRank']

# Applicare One-Hot Encoding alle colonne categoriche
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Specifica il percorso di output del nuovo file CSV
output_path = "../../dataset/finalDataset_encoded.csv"

# Salvare il dataset trasformato in un nuovo file CSV
df_encoded.to_csv(output_path, index=False)


