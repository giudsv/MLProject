import duckdb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Connessione al database DuckDB
con = duckdb.connect("../dataset/database/ufc_predictions.db")

# Recupero dei dati dal database
query = "SELECT * FROM fights"  # Sostituisci con il nome della tua tabella
df = con.execute(query).fetchdf()

# Visualizziamo i primi dati per capire la struttura
print(df.head())

# Pre-elaborazione dei dati
#'Winner' la variabile target e tutte le altre colonne siano variabili indipendenti
X = df.drop('Winner', axis=1)  # Colonne indipendenti
y = df['Winner']  # Variabile target (dipendente)

# Suddividere i dati in training e test (80%-20%)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creazione del modello Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Alleniamo il modello
rf.fit(X_train, y_train)

# Prediciamo con i dati di test
y_pred = rf.predict(X_test)

# Calcoliamo la precisione
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy del modello Random Forest: {accuracy:.2f}")

# Opzionale: Salvare il modello addestrato
import joblib
joblib.dump(rf, 'random_forest_model.pkl')
