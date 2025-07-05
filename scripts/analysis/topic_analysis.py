import pandas as pd
import os
import config_analysis as config
import openai

#openai.api_key = API_KEY

def get_party(username):
    if username in config.afd_usernames:
        return "AfD"
    elif username in config.spd_usernames:
        return "SPD"
    elif username in config.cdu_csu_usernames:
        return "CDU_CSU"
    elif username in config.gruene_usernames:
        return "Grüne"
    elif username in config.linke_usernames:
        return "Linke"

def gpt_topic_classification(description, transcript, party):
    prompt = (
        "Ordne das folgende TikTok Video in einen dieser Themenbereiche ein: Soziales & Arbeit, Wirtschaft & Finanzen, Sicherheit & Ordnung, Migration, Umwelt & Energie, Internationale Politik, Persönliches, Wahlkampf.\n"
        "Gib nur den ausgewählten Themenbereich zurück, ohne Erklärung.\n"
        f"Beschreibung: {description}\n"
        f"Transkript: {transcript}\n"
        f"Partei: {party}\n"
    )
    print(prompt)
    try:
        response = openai.Completion.create(
            model="ft:gpt-4.1-nano-2025-04-14:caro-haensch:reginav2:BoWObdmF",
            prompt=prompt,
            max_tokens=20,
            temperature=0.0
        )
        return response.choices[0].text
    except Exception as e:
        print(f"Fehler bei Anfrage: {e}")
        return("ERROR")

def main():
    folder = "data/data_preprocessed/videos"
    all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]
    dfs = []
    for file in all_files:
        df = pd.read_csv(file, engine='python', on_bad_lines='warn')
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