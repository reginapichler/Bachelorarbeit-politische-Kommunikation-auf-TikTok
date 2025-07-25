import pandas as pd
import os
import config_analysis as config
import openai

# set API key for OpenAI here
openai.api_key = API_KEY

def get_party(username):
    """Returns the party of the user based on their username."""
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
    """Send request to fine-tuned GPT-4.1-nano model to classify the topic"""
    # define the prompt
    prompt = (
        "Ordne das folgende TikTok Video in einen dieser Themenbereiche ein: Soziales & Arbeit, Wirtschaft & Finanzen, Sicherheit & Ordnung, Migration, Umwelt & Energie, Internationale Politik, Persönliches, Wahlkampf.\n"
        "Gib nur den ausgewählten Themenbereich zurück, ohne Erklärung.\n"
        f"Beschreibung: {description}\n"
        f"Transkript: {transcript}\n"
        f"Partei: {party}\n"
    )
    print(prompt)
    try:
        # send request
        response = openai.Completion.create(
            model="ft:gpt-4.1-nano-2025-04-14:caro-haensch:reginav2:BoWObdmF",
            prompt=prompt,
            max_tokens=10,
            temperature=0.0
        )
        return response.choices[0].text
    except Exception as e:
        print(f"Error for query: {e}")
        return("ERROR")

def main():
    # configurations
    folder = "data/data_preprocessed/videos"
    all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]
    dfs = []
    # get videos of each user and assign party
    for file in all_files:
        df = pd.read_csv(file, engine='python', on_bad_lines='warn')
        basename = os.path.basename(file)
        username = basename.split("_video_data")[0]
        df["username"] = username
        df["party"] = df["username"].apply(get_party)
        dfs.append(df)
    all_videos = pd.concat(dfs, ignore_index=True)
    all_videos = all_videos[["id", "username", "party", "video_description", "voice_to_text"]]

    output_dir = f"results/topic_analysis/{config.start_date}_{config.end_date}"
    os.makedirs(output_dir, exist_ok=True)

    # process videos of each party
    for party in all_videos["party"].unique():
        party_df = all_videos[all_videos["party"] == party].copy()
        results = []
        for idx, row in party_df.iterrows():
            description = str(row["video_description"])
            transcript = str(row["voice_to_text"])
            # get topic classification
            gpt_response = gpt_topic_classification(description, transcript, party)
            results.append(gpt_response)
        party_df["gpt_topic"] = results
        party_df.to_csv(os.path.join(output_dir, f"{party}.csv"), index=False)
        print(f"Done: {party}")

if __name__ == "__main__":
    main()