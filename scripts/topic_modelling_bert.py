import pandas as pd
import os
import config
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords

video_dir = os.path.join("data", "data_preprocessed", "videos")
results_dir = os.path.join("results", "topic_modelling", "bert")
os.makedirs(results_dir, exist_ok=True)

parteien = {
    "Linke": config.linke_usernames,
    "Grüne": config.gruene_usernames,
    "CDUCSU": config.cdu_csu_usernames,
    "AfD": config.afd_usernames,
    "SPD": config.spd_usernames
}
start_date = config.start_date
end_date = config.end_date

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

stop_words = set(stopwords.words('german'))
eigene_stopwords = {'ja', 'äh', 'ne'}  # ggf. erweitern
stop_words.update(eigene_stopwords)

print("Stopwörter:", stop_words)

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

for partei, userlist in parteien.items():
    dfs = []
    for user in userlist:
        file_path = os.path.join(video_dir, f"{user}_video_data_preprocessed_{start_date}_{end_date}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dfs.append(df)
    if not dfs:
        print(f"Keine Daten für {partei}")
        continue

    partei_df = pd.concat(dfs, ignore_index=True)
    partei_df['text'] = partei_df['video_description'].fillna('') + " " + partei_df['voice_to_text'].fillna('')
    partei_df['text'] = partei_df['text'].astype(str).apply(remove_stopwords)
    texts = partei_df['text'].astype(str).tolist()

    # Embeddings berechnen
    embeddings = model.encode(texts, show_progress_bar=True)

    # BERTopic-Modell
    topic_model = BERTopic(language="multilingual", calculate_probabilities=True, verbose=True)
    topics, probs = topic_model.fit_transform(texts, embeddings)

    # Top Words pro Topic extrahieren
    topics_info = topic_model.get_topic_info()
    topics_info.to_csv(os.path.join(results_dir, f"{partei}_{start_date}_{end_date}_bertopic_topics.csv"), index=False)

    # Optional: Top 5 Topics ausgeben
    print(f"\nTop Topics für {partei}:")
    print(topics_info.head())