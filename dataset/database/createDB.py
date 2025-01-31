import duckdb
import pandas as pd

# Carica il dataset in Pandas
df = pd.read_csv("../../dataset/finalDataset.csv")

# Crea un database DuckDB e una tabella
con = duckdb.connect("ufc_predictions.db")
con.execute("CREATE TABLE IF NOT EXISTS fights AS SELECT * FROM df")

# Controlla se i dati sono stati caricati
result = con.execute("SELECT COUNT(*) FROM fights").fetchall()
print(f"Righe caricate: {result[0][0]}")
