import pandas as pd

# Caricare il dataset
df = pd.read_csv('../../dataset/finalDataset.csv')

# Ordinare i dati per data del match (DaysSinceFirstFight) per processarli in ordine temporale
df = df.sort_values(by='DaysSinceFirstFight').reset_index(drop=True)

# Creiamo due nuove colonne per il numero di giorni trascorsi dall'ultimo incontro
df['DaysSinceLastFight_Red'] = None
df['DaysSinceLastFight_Blue'] = None

# Dizionari per memorizzare l'ultimo combattimento di ogni fighter
last_fight_dict = {}

# Iteriamo sulle righe per calcolare i giorni dall'ultimo fight
for index, row in df.iterrows():
    red_fighter = row['RedFighter']   # Nome atleta rosso
    blue_fighter = row['BlueFighter'] # Nome atleta blu
    fight_date = row['DaysSinceFirstFight']  # Giorni trascorsi dal 2010-03-21

    # Giorni dall'ultimo fight per l'atleta rosso
    if red_fighter in last_fight_dict:
        df.at[index, 'DaysSinceLastFight_Red'] = fight_date - last_fight_dict[red_fighter]
    else:
        df.at[index, 'DaysSinceLastFight_Red'] = None  # Nessun fight precedente

    # Giorni dall'ultimo fight per l'atleta blu
    if blue_fighter in last_fight_dict:
        df.at[index, 'DaysSinceLastFight_Blue'] = fight_date - last_fight_dict[blue_fighter]
    else:
        df.at[index, 'DaysSinceLastFight_Blue'] = None  # Nessun fight precedente

    # Aggiorniamo il dizionario con l'ultima data di combattimento per ogni fighter
    last_fight_dict[red_fighter] = fight_date
    last_fight_dict[blue_fighter] = fight_date

# Convertiamo le colonne in numerico, riempiendo i NaN con 0 (per gli atleti al primo match)
df['DaysSinceLastFight_Red'] = pd.to_numeric(df['DaysSinceLastFight_Red'], errors='coerce').fillna(0)
df['DaysSinceLastFight_Blue'] = pd.to_numeric(df['DaysSinceLastFight_Blue'], errors='coerce').fillna(0)

# Salvare il dataset aggiornato
df.to_csv('../../dataset/finalDataset.csv', index=False)


