import pandas as pd
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score

# Caricare il dataset per il mapping dei fighter
df_fighters = pd.read_csv("../../dataset/finalDataset_encoded.csv")
print(df_fighters['Winner_Red'].value_counts())

fighters_mapping = pd.read_csv("../../dataset/fighters_labels.csv")
fighters_dict = dict(zip(fighters_mapping["Fighter"].str.lower().str.strip(), fighters_mapping["Label"]))


# Funzione per allenare il modello
def train_model():
    df = df_fighters.sort_values(by="DaysSinceFirstFight")
    X = df.drop('Winner_Red', axis=1)
    y = df['Winner_Red']

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=100, n_jobs=-1, class_weight='balanced')

    method = input("Scegli il metodo di training (1 per TimeSeriesSplit, 2 per 80-20 split): ")

    if method == "1":
        print("\n▶ Training con TimeSeriesSplit...")
        tscv = TimeSeriesSplit(n_splits=5)
        accuracy_scores, f1_scores = [], []

        start_time = time.time()

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            rf_classifier.fit(X_train, y_train)
            y_pred = rf_classifier.predict(X_test)

            accuracy_scores.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))

        execution_time = time.time() - start_time
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

        start_time = time.time()

        rf_classifier.fit(X_train, y_train)
        y_pred = rf_classifier.predict(X_test)

        execution_time = time.time() - start_time

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

# Funzione per fare una predizione
def predict_model():
    df = df_fighters.sort_values(by="DaysSinceFirstFight")
    X = df.drop('Winner_Red', axis=1)

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

    red_fighter = input("Inserisci il nome del Red Fighter: ").strip().lower()
    blue_fighter = input("Inserisci il nome del Blue Fighter: ").strip().lower()

    if red_fighter not in fighters_dict or blue_fighter not in fighters_dict:
        print("❌ Uno dei nomi inseriti non è presente nel database. Riprova.")
        return

    red_fighter_id = fighters_dict[red_fighter]
    blue_fighter_id = fighters_dict[blue_fighter]

    red_fighter_stats = df[(df['RedFighter'] == red_fighter_id) | (df['BlueFighter'] == red_fighter_id)]
    blue_fighter_stats = df[(df['RedFighter'] == blue_fighter_id) | (df['BlueFighter'] == blue_fighter_id)]

    if red_fighter_stats.empty or blue_fighter_stats.empty:
        print("❌ Errore: uno dei fighter non è presente nel dataset.")
        return

    fight_data = pd.DataFrame(columns=X.columns)

    for col in red_fighter_stats.columns:
        if col in fight_data.columns:
            fight_data.at[0, col] = red_fighter_stats[col].values[0]

    for col in blue_fighter_stats.columns:
        if col in fight_data.columns:
            fight_data.at[0, col] = blue_fighter_stats[col].values[0]

    expected_features = rf_classifier.feature_names_in_
    fight_data = fight_data.reindex(columns=expected_features, fill_value=0)

    start_time = time.time()
    prediction = rf_classifier.predict(fight_data)[0]
    execution_time = time.time() - start_time

    winner = red_fighter if prediction == 1 else blue_fighter
    print(f"\n🥊 Il modello prevede che il vincitore sarà: **{winner}** 🏆")
    print(f"⏳ Tempo di esecuzione (solo predizione): {execution_time:.2f} secondi")

    print("\n🔍 Controllo output modello (conteggio previsioni):")
    print(pd.Series(prediction).value_counts())

    prediction_proba = rf_classifier.predict_proba(fight_data)
    print(f"Prediction probabilities: {prediction_proba}")
    print(f"Raw prediction: {prediction}")


# Menu principale
if __name__ == "__main__":
    mode = input("Scegli modalità: train (1) o predict (2): ")
    if mode == "1":
        train_model()
    elif mode == "2":
        predict_model()
    else:
        print("❌ Scelta non valida.")
