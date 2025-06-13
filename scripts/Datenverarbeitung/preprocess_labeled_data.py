import os
import glob
import pandas as pd
import csv
import json
import random

folder = "data/data_raw/videos/labeled"
output_folder = "data/data_preprocessed/videos/labeled"
os.makedirs(output_folder, exist_ok=True)
pattern = os.path.join(folder, "*.csv")
files = glob.glob(pattern)

dfs = []
for file in files:
    print(f"Lade Datei: {file}")
    df = pd.read_csv(
        file,
        sep=";",
        encoding="utf-8",
        engine="python",
        on_bad_lines="warn"
    )
    # Leere Strings und leere Listen als NA setzen
    for col in df.columns:
        df[col] = df[col].replace('', pd.NA)
        df[col] = df[col].replace('[]', pd.NA)
    # Speichern
    basename = os.path.basename(file)
    output_path = os.path.join(output_folder, basename.replace(".csv", "_preprocessed.csv"))
    df.to_csv(output_path, index=False)
    dfs.append(df)

if dfs:

    for i, df in enumerate(dfs):
        print(f"Datei {i} Spalten: {df.columns.tolist()}")

    # Alle DataFrames auf die gleiche Spaltenreihenfolge bringen

    desired_columns = ["id", "voice_to_text", "video_description", "Topic"]

    all_data = pd.concat(dfs, ignore_index=True)

    # Alles zu String machen, fehlende Werte als "None" setzen
    all_data["Topic"] = all_data["Topic"].astype(str)
    all_data["Topic"] = all_data["Topic"].replace({"nan": "None", "None": "None"})

    print(f"Anzahl aller gelabelten Einträge: {len(all_data)}")
    print("\nAnzahl pro Topic-Kategorie:")
    print(all_data["Topic"].value_counts(dropna=False))
    print(all_data.loc[all_data["Topic"].isna() | (all_data["Topic"].str.lower() == "none"), ["id", "Topic"]])
    print(all_data["Topic"].apply(type).value_counts())

    # Speichere alle Zeilen ohne Topic in eine CSV
    no_topic = all_data[all_data["Topic"].str.lower() == "none"]
    no_topic.to_csv("data/data_raw/no_topic_entries.csv", index=False)

    # --- JSONL-Export für 20 zufällige Beispiele pro Topic ---
    jsonl_path = "data/data_raw/videos/labeled/topic_examples.jsonl"
    topics = ['W', 'W&F', 'S&O', 'M', 'U&E', 'I', 'P']
    system_prompt = (
        "Du bist ein Politikwissenschaftler, der TikTok Videobeschreibungen und -transkripte in folgende Kategorien einordnet: "
        "Soziales & Arbeit, Wirtschaft & Finanzen, Sicherheit & Ordnung, Migration, Umwelt & Energie, Internationale Politik, "
        "Persönliches, Wahlkampf. Wenn mehrere Themen angesprochen werden, gebe die zwei dominantesten Bereiche aus. "
        "Deine Antwort besteht nur aus dem/den gewählten Themenbereich(en) für die jeweiligen Texte."
    )

    # Mapping der Topics auf ausgeschriebene Begriffe
    topic_mapping = {
        'W': 'Wahlkampf',
        'W&F': 'Wirtschaft & Finanzen',
        'S&O': 'Sicherheit & Ordnung',
        'M': 'Migration',
        'U&E': 'Umwelt & Energie',
        'I': 'Internationale Politik',
        'P': 'Persönliches'
    }

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for topic in topics:
            if topic == "None":
                continue
            df_topic = all_data[all_data["Topic"] == topic]
            if len(df_topic) > 20:
                df_topic = df_topic.sample(20, random_state=42)
            for _, row in df_topic.iterrows():
                user_content = f"Beschreibung: {row['video_description']}; Transkript: {row['voice_to_text']}"
                # Mapping anwenden:
                assistant_content = topic_mapping.get(row["Topic"], row["Topic"])
                entry = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content}
                    ]
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nJSONL-Datei mit Beispielen pro Topic gespeichert unter: {jsonl_path}")
else:
    print("Keine gelabelten Dateien gefunden.")

