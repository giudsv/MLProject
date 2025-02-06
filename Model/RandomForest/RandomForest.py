import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# Caricare il dataset
df = pd.read_csv('../../dataset/finalDataset_encoded.csv')  # Sostituisci con il tuo file CSV

# Pre-elaborazione dei dati
X = df.drop('Winner_Red', axis=1)  # Colonne indipendenti (rimuoviamo la variabile target)
y = df['Winner_Red']  # Variabile target (continua o binaria, dipende dai dati)

# Creazione del modello Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Eseguiamo la 5-fold cross-validation
cv_scores_accuracy = cross_val_score(rf_classifier, X, y, cv=5, scoring='accuracy')  # Accuracy per la classificazione

# Calcoliamo la precisione media delle 5 fold
mean_accuracy = np.mean(cv_scores_accuracy)

# Visualizziamo i risultati
print(f"Mean Accuracy (5-fold CV): {mean_accuracy:.2f}")

# Opzionale: Allenare su tutto il dataset e salvare il modello addestrato
rf_classifier.fit(X, y)  # Allenamento su tutto il dataset
joblib.dump(rf_classifier, 'random_forest_classifier_model.pkl')  # Salvataggio del modello

print("Modello Random Forest Classifier salvato come 'random_forest_classifier_model.pkl'")
