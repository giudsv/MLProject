import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np

# Caricare il dataset
df = pd.read_csv('../../dataset/finalDataset_encoded.csv')  # Sostituisci con il percorso corretto del tuo file CSV

# Pre-elaborazione dei dati
X = df.drop('Winner_Red', axis=1)  # Colonne indipendenti
y = df['Winner_Red']  # Variabile target continua

# Creazione del modello di regressione lineare
lr_model = LinearRegression()

# Eseguiamo la k-fold cross-validation (ad esempio 5 fold)
cv_scores = cross_val_score(lr_model, X, y, cv=5, scoring='neg_mean_squared_error')

# Calcoliamo l'errore quadratico medio (MSE) medio delle k fold
mean_mse = np.mean(-cv_scores)  # Invertiamo il segno, poiché 'cross_val_score' restituisce valori negativi per MSE
mean_r2 = np.mean(cross_val_score(lr_model, X, y, cv=5, scoring='r2'))

# Visualizziamo i risultati
print(f"Mean Squared Error (MSE) medio (5-fold CV): {mean_mse:.2f}")
print(f"Mean R^2 Score medio (5-fold CV): {mean_r2:.2f}")
