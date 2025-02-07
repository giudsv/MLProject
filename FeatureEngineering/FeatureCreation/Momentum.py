import pandas as pd

# Caricare il dataset
df = pd.read_csv('../../dataset/finalDataset.csv')

# Ordinare il dataset per data del fight (DaysSinceFirstFight)
df = df.sort_values(by='DaysSinceFirstFight')

# Dizionario per tracciare la storia dei match con date
momentum_dict = {}

# Inizializziamo le nuove colonne
df['Momentum_Red'] = 0
df['Momentum_Blue'] = 0

# Iteriamo su ogni match
for index, row in df.iterrows():
    red_fighter = row['RedFighter']
    blue_fighter = row['BlueFighter']
    weight_class = row['WeightClass']
    winner = row['Winner']
    fight_date = row['DaysSinceFirstFight']

    red_last_fight_days = row['DaysSinceLastFight_Red']
    blue_last_fight_days = row['DaysSinceLastFight_Blue']

    red_rank = row['RMatchWCRank']
    blue_rank = row['BMatchWCRank']

    # Se il fighter non ha incontri precedenti, inizializziamo la lista vuota
    if red_fighter not in momentum_dict:
        momentum_dict[red_fighter] = []
    if blue_fighter not in momentum_dict:
        momentum_dict[blue_fighter] = []


    # Funzione per calcolare il momentum con massimo 5 incontri e distanza massima di 150 giorni
    def calculate_momentum(fighter, last_fight_days):
        if fighter not in momentum_dict:
            return 0

        past_fights = momentum_dict[fighter]

        # Filtriamo solo gli ultimi 5 match entro 150 giorni dal precedente fight
        valid_fights = []
        for past_score, past_date in reversed(past_fights):  # Partiamo dagli incontri più recenti
            if len(valid_fights) >= 5:
                break  # Abbiamo già i 5 incontri validi
            if last_fight_days - past_date > 150:
                break  # Se il match è troppo lontano, ci fermiamo

            valid_fights.append((past_score, past_date))

        return sum(fight[0] for fight in valid_fights)


    # Calcoliamo il momentum per entrambi i fighter
    df.at[index, 'Momentum_Red'] = calculate_momentum(red_fighter, red_last_fight_days)
    df.at[index, 'Momentum_Blue'] = calculate_momentum(blue_fighter, blue_last_fight_days)

    # Ora aggiorniamo il momentum con il match corrente
    if winner == 'Red':
        if red_rank > blue_rank:  # Red ha battuto un fighter di rank maggiore
            score_red = +2
            score_blue = -1
        else:  # Red ha battuto un fighter di rank minore
            score_red = +1
            score_blue = -2
    elif winner == 'Blue':
        if blue_rank > red_rank:  # Blue ha battuto un fighter di rank maggiore
            score_blue = +2
            score_red = -1
        else:  # Blue ha battuto un fighter di rank minore
            score_blue = +1
            score_red = -2
    else:  # Pareggio
        score_red = 0
        score_blue = 0

    # Aggiorniamo il dizionario con il nuovo risultato
    momentum_dict[red_fighter].append((score_red, fight_date))
    momentum_dict[blue_fighter].append((score_blue, fight_date))

# Salviamo il dataset con la nuova feature
df.to_csv('../../dataset/finalDataset.csv', index=False)

print("✅ Momentum calcolato correttamente con il filtro")
