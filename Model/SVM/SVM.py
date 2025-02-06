import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import joblib

# Carica il dataset
df = pd.read_csv('../../dataset/finalDataset_encoded.csv')  # Modifica con il percorso corretto del tuo file

# Pre-elaborazione dei dati
X = df.drop('Winner_Red', axis=1)  # Colonne indipendenti
y = df['Winner_Red']  # Variabile target (Winner_Red)

# Creazione del modello SVM con kernel RBF (Radial Basis Function)
svm_model = SVC(kernel='rbf', random_state=42)

# Applicazione della 5-fold cross-validation
cv_scores = cross_val_score(svm_model, X, y, cv=5, scoring='accuracy')

# Stampa dell'accuratezza media delle 5 fold
print(f"Accuracy media con 5-fold Cross Validation: {cv_scores.mean():.2f}")

# Opzionale: Salvare il modello addestrato (su tutto il dataset)
svm_model.fit(X, y)  # Alleniamo il modello con l'intero dataset
joblib.dump(svm_model, 'svm_model.pkl')

print("Modello SVM salvato come 'svm_model.pkl'")
