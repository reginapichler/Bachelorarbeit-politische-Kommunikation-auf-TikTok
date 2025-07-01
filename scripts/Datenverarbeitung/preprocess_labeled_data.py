import os
import glob
import pandas as pd
import csv
import json
import random
import config_processing as config

folder = "data/data_raw/videos/labeled"
output_folder = "data/data_preprocessed/videos/labeled"
os.makedirs(output_folder, exist_ok=True)
pattern = os.path.join(folder, "*.csv")
files = glob.glob(pattern)

def get_party(account):
        # Passe die Zuordnung an deine config_preprocessing-Struktur an!
        if account in config.cdu_csu_usernames:
            return "CDU/CSU"
        elif account in config.afd_usernames:
            return "AfD"
        elif account in config.gruene_usernames:
            return "Bündnis 90/Die Grünen"
        elif account in config.spd_usernames:
            return "SPD"
        elif account in config.linke_usernames:
            return "Die Linke"

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
    # Accountname aus Dateiname extrahieren
    basename = os.path.basename(file)
    account = basename.split("_video_data")[0]
    df["partei"] = get_party(account)
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
    topics = ['S&A', 'W', 'W&F', 'S&O', 'M', 'U&E', 'I', 'P']
    system_prompt = (
       "Du bist ein Politikwissenschaftler, der Inhalte von TikTok Videos deutscher Parteien in folgende Themenbereiche einordnet: Soziales & Arbeit, Wirtschaft & Finanzen, Sicherheit & Ordnung, Migration, Umwelt & Energie, Internationale Politik, Persönliches, Wahlkampf. Wenn mehrere Themen angesprochen werden, gebe die zwei dominantesten Bereiche aus. Deine Antwort besteht nur aus dem/den gewählten Themenbereich(en) für den vorliegenden Text. Dir sind die Videobeschreibung, das Transkript des gesprochenen Inhaltes und die Partei der Aussage gegeben."
    )

    # Mapping der Topics auf ausgeschriebene Begriffe
    topic_mapping = {
        'S%A': 'Soziales & Arbeit',
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
            df_topic = all_data[all_data["Topic"] == topic]

            # bist zu 4 zufällige Beispiele pro Partei auswählen
            df_list = []
            for partei, group in df_topic.groupby("partei"):
                n = min(len(group), 4)
                df_list.append(group.sample(n, random_state=42))
                print(f"Anzahl Beispiele für {partei} im Topic {topic}: {n}")
            df_topic_sampled = pd.concat(df_list)

            # falls weniger als 20 Beispiele: auffüllen
            if len(df_topic_sampled) < 20:
                rest = df_topic.drop(df_topic_sampled.index)
                n_rest = 20 - len(df_topic_sampled)
                print(f"Topic {topic} hat weniger als 20 Beispiele, füge {n_rest} weitere zufällige hinzu.")
                if len(rest) > 0:
                    df_topic_sampled = pd.concat([df_topic_sampled, rest.sample(min(n_rest, len(rest)), random_state=42)])

            print(f"Anzahl Beispiele für Topic {topic}: {len(df_topic_sampled)}")
            
            for _, row in df_topic_sampled.iterrows():
                user_prompt = f"Beschreibung: {row['video_description']}; Transkript: {row['voice_to_text']}; Partei: {row['partei']}"
                assistant_content = topic_mapping.get(row["Topic"], row["Topic"])
                entry = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant_content}
                    ]
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nJSONL-Datei mit Beispielen pro Topic gespeichert unter: {jsonl_path}")
else:
    print("Keine gelabelten Dateien gefunden.")

