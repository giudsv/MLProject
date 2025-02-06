import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Caricare il dataset
df = pd.read_csv("../../dataset/finalDataset.csv")

# Creare un oggetto LabelEncoder
label_encoder = LabelEncoder()

# Applicare Label Encoding agli atleti (RedFighter e BlueFighter) e alla Location
df['RedFighter'] = label_encoder.fit_transform(df['RedFighter'])
df['BlueFighter'] = label_encoder.fit_transform(df['BlueFighter'])
df['Location'] = label_encoder.fit_transform(df['Location'])
df['Country'] = label_encoder.fit_transform(df['Country'])


# Selezioniamo le colonne categoriche per One-Hot Encoding
categorical_columns = ['BlueStance', 'RedStance', 'Winner', 'TitleBout', 'WeightClass', 'Gender', 'Finish', 'BetterRank']

# Applicare One-Hot Encoding alle altre colonne categoriche
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Specifica il percorso di output del nuovo file CSV
output_path = "../../dataset/finalDataset_encoded.csv"

# Salvare il dataset trasformato in un nuovo file CSV
df_encoded.to_csv(output_path, index=False)

print("Label Encoding per gli atleti e Location e One-Hot Encoding per le altre colonne completati. File salvato come finalDataset_encoded.csv.")

