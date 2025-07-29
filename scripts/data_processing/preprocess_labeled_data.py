import os
import glob
import pandas as pd
import json
import config_processing as config

def get_party(account):
    """Get party by username."""
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
        
def main():
    # configurations
    folder = "data/data_raw/videos/labeled"
    output_folder = "data/data_preprocessed/videos/labeled"
    os.makedirs(output_folder, exist_ok=True)
    pattern = os.path.join(folder, "*.csv")
    # get all labeled video files
    files = glob.glob(pattern)

    dfs = []
    for file in files:
        print(f"Load file: {file}")
        df = pd.read_csv(
            file,
            sep=";",
            encoding="utf-8",
            engine="python",
            on_bad_lines="warn"
        )
        # get account name from filename
        basename = os.path.basename(file)
        account = basename.split("_video_data")[0]
        df["party"] = get_party(account)
        # set empty strings and empty lists to NaN
        for col in df.columns:
            df[col] = df[col].replace('', pd.NA)
            df[col] = df[col].replace('[]', pd.NA)
        # save
        basename = os.path.basename(file)
        output_path = os.path.join(output_folder, basename.replace(".csv", "_preprocessed.csv"))
        df.to_csv(output_path, index=False)
        dfs.append(df)

    if dfs:
        all_data = pd.concat(dfs, ignore_index=True)

        # convert data types
        all_data["Topic"] = all_data["Topic"].astype(str)
        all_data["Topic"] = all_data["Topic"].replace({"nan": "None", "None": "None"})

        print(f"Number of labeled entries {len(all_data)}")
        print("\nNumber of entries per category:")
        print(all_data["Topic"].value_counts(dropna=False))
        # show all entries without topic
        print(all_data.loc[all_data["Topic"].isna() | (all_data["Topic"].str.lower() == "none"), ["id", "Topic"]])
        # show datatypes of topic entries
        print(all_data["Topic"].apply(type).value_counts())

        # convert to jsonl format for LLM training
        # create jsonl file with 20 examples per topic
        # choose up to 4 examples per party for each topic
        jsonl_path = "data/data_raw/videos/labeled/topic_examples.jsonl"
        # define topic shortnames
        topics = ['S&A', 'W', 'W&F', 'S&O', 'M', 'U&E', 'I', 'P']
        system_prompt = (
        "Du bist ein Politikwissenschaftler, der Inhalte von TikTok Videos deutscher Parteien in folgende Themenbereiche einordnet: Soziales & Arbeit, Wirtschaft & Finanzen, Sicherheit & Ordnung, Migration, Umwelt & Energie, Internationale Politik, Persönliches, Wahlkampf. Wenn mehrere Themen angesprochen werden, gebe die zwei dominantesten Bereiche aus. Deine Antwort besteht nur aus dem/den gewählten Themenbereich(en) für den vorliegenden Text. Dir sind die Videobeschreibung, das Transkript des gesprochenen Inhaltes und die Partei der Aussage gegeben."
        )

        # mapping for the topics to human-readable labels
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

        # create jsonl
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for topic in topics:
                # get entries for respecting topic
                df_topic = all_data[all_data["Topic"] == topic]
                # up to 4 examples per party
                df_list = []
                for party, group in df_topic.groupby("party"):
                    n = min(len(group), 4)
                    df_list.append(group.sample(n, random_state=42))
                    print(f"Number of examples from {party} for topic {topic}: {n}")
                df_topic_sampled = pd.concat(df_list)

                # filling the rest with random samples if less than 4 examples per party
                if len(df_topic_sampled) < 20:
                    rest = df_topic.drop(df_topic_sampled.index)
                    n_rest = 20 - len(df_topic_sampled)
                    print(f"Topic {topic} has less than 20 examples, add {n_rest} more random examples.")
                    if len(rest) > 0:
                        df_topic_sampled = pd.concat([df_topic_sampled, rest.sample(min(n_rest, len(rest)), random_state=42)])
 
                print(f"Number of examples for topic {topic}: {len(df_topic_sampled)}")
                
                # create jsonl entries
                for _, row in df_topic_sampled.iterrows():
                    user_prompt = f"Beschreibung: {row['video_description']}; Transkript: {row['voice_to_text']}; party: {row['party']}"
                    assistant_content = topic_mapping.get(row["Topic"], row["Topic"])
                    entry = {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": assistant_content}
                        ]
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\nJSONL-file with examples saved at: {jsonl_path}")
        # Note: the jsonl file has been modified manually afterwards by adding \ in front of all " to avoid errors at processing.
    else:
        print("No labeled data found.")

if __name__ == "__main__":
    main()