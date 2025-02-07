import pandas as pd
import numpy as np
import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score


# Funzione per allenare il modello
def train_model():
    df = pd.read_csv('../../dataset/finalDataset_encoded.csv')  # Sostituisci con il tuo file CSV
    df = df.sort_values(by="DaysSinceFirstFight")
    X = df.drop('Winner_Red', axis=1)
    y = df['Winner_Red']

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    method = input("Scegli il metodo di training (1 per TimeSeriesSplit, 2 per 80-20 split): ")

    if method == "1":
        print("\n▶ Training con TimeSeriesSplit...")
        tscv = TimeSeriesSplit(n_splits=5)
        accuracy_scores, f1_scores = [], []

        start_time = time.time()  # Inizia il conteggio per il training

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            rf_classifier.fit(X_train, y_train)
            y_pred = rf_classifier.predict(X_test)

            accuracy_scores.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))

        execution_time = time.time() - start_time  # Fine del conteggio

        mean_accuracy = np.mean(accuracy_scores)
        mean_f1_score = np.mean(f1_scores)
        print(f"✅ Mean Accuracy (TimeSeriesSplit CV): {mean_accuracy:.4f}")
        print(f"✅ Mean F1-score (TimeSeriesSplit CV): {mean_f1_score:.4f}")
        model_filename = 'random_forest_K_CROSS.pkl'

    elif method == "2":
        print("\n▶ Training con 80-20 split...")
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        start_time = time.time()  # Inizia il conteggio per il training

        rf_classifier.fit(X_train, y_train)
        y_pred = rf_classifier.predict(X_test)

        execution_time = time.time() - start_time  # Fine del conteggio

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"✅ Accuracy: {accuracy:.4f}")
        print(f"✅ F1-score: {f1:.4f}")
        model_filename = 'random_forest_SPLIT.pkl'
    else:
        print("❌ Scelta non valida.")
        return

    print(f"⏳ Tempo di esecuzione (solo training): {execution_time:.2f} secondi")
    joblib.dump(rf_classifier, model_filename)
    print(f"💾 Modello salvato come {model_filename}")


# Funzione per fare previsioni
def predict_model():
    df = pd.read_csv('../../dataset/finalDataset_encoded.csv')
    df = df.sort_values(by="DaysSinceFirstFight")
    X = df.drop('Winner_Red', axis=1)
    y = df['Winner_Red']

    model_choice = input("Scegli il modello da caricare (1 per K-CROSS, 2 per SPLIT): ")

    if model_choice == "1":
        model_filename = 'random_forest_K_CROSS.pkl'
    elif model_choice == "2":
        model_filename = 'random_forest_SPLIT.pkl'
    else:
        print("❌ Scelta non valida.")
        return

    rf_classifier = joblib.load(model_filename)
    print(f"✅ Modello {model_filename} caricato con successo!")

    start_time = time.time()  # Inizia il conteggio solo per la predizione

    y_pred = rf_classifier.predict(X)

    execution_time = time.time() - start_time  # Fine del conteggio

    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"\n🎯 Accuracy: {accuracy:.4f}")
    print(f"🎯 F1-score: {f1:.4f}")
    print(f"⏳ Tempo di esecuzione (solo predizione): {execution_time:.2f} secondi")


# Menu principale
if __name__ == "__main__":
    mode = input("Scegli modalità: train (1) o predict (2): ")
    if mode == "1":
        train_model()
    elif mode == "2":
        predict_model()
    else:
        print("❌ Scelta non valida.")
