import pandas as pd
import os
import requests
import config_analysis as config

# Lade API-Key und Endpoint
with open("key.txt", "r") as f:
    API_KEY = f.read().strip()
with open("endpoint.txt", "r") as f:
    ENDPOINT = f.read().strip()

system_prompt = (
    "Du bist ein Politikwissenschaftler, der Inhalte von TikTok Videos deutscher Parteien in folgende Themenbereiche einordnet: "
    "Soziales & Arbeit, Wirtschaft & Finanzen, Sicherheit & Ordnung, Migration, Umwelt & Energie, Internationale Politik, Persönliches, Wahlkampf. "
    "Wenn mehrere Themen angesprochen werden, gebe die zwei dominantesten Bereiche aus. "
    "Deine Antwort besteht nur aus dem/den gewählten Themenbereich(en) für den vorliegenden Text. "
    "Dir sind die Videobeschreibung, das Transkript des gesprochenen Inhaltes und die Partei der Aussage gegeben."
)

def get_party(username):
    if username in config.afd_usernames:
        return "AfD"
    elif username in config.spd_usernames:
        return "SPD"
    elif username in config.cdu_csu_usernames:
        return "CDU/CSU"
    elif username in config.gruene_usernames:
        return "Grüne"
    elif username in config.linke_usernames:
        return "Linke"

def gpt_topic_classification(description, transcript, party):
    user_prompt = f"Beschreibung: {description}; Transkript: {transcript}; Partei: {party}"
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 100,
        "top_p": 0.95
    }
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }
    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Fehler bei Anfrage: {e}")
        return "ERROR"


def main():
    folder = "data/data_raw/videos"
    all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]
    dfs = []
    for file in all_files:
        df = pd.read_csv(file)
        basename = os.path.basename(file)
        username = basename.split("_video_data")[0]
        df["username"] = username
        df["partei"] = df["username"].apply(get_party)
        dfs.append(df)
    all_videos = pd.concat(dfs, ignore_index=True)
    all_videos = all_videos[["id", "username", "partei", "video_description", "voice_to_text"]]

    output_dir = f"results/topic_analysis/{config.start_date}_{config.end_date}"
    os.makedirs(output_dir, exist_ok=True)

    for partei in all_videos["partei"].unique():
        partei_df = all_videos[all_videos["partei"] == partei].copy()
        results = []
        for idx, row in partei_df.iterrows():
            description = str(row["video_description"])
            transcript = str(row["voice_to_text"])
            gpt_response = gpt_topic_classification(description, transcript, partei)
            results.append(gpt_response)
        partei_df["gpt_topic"] = results
        partei_df.to_csv(os.path.join(output_dir, f"{partei}.csv"), index=False)
        print(f"Fertig: {partei}")

if __name__ == "__main__":
    main()