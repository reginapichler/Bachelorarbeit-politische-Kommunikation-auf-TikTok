import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
import scripts.data_processing.config_processing as config_processing
from nltk.corpus import stopwords

video_dir = os.path.join("data", "data_preprocessed", "videos")
results_dir = os.path.join("results", "topic_modelling", "lda")

parteien = {
    "Linke": config_processing.linke_usernames,
    "Grüne": config_processing.gruene_usernames,
    "CDUCSU": config_processing.cdu_csu_usernames,
    "AfD": config_processing.afd_usernames,
    "SPD": config_processing.spd_usernames
}
start_date = config_processing.start_date
end_date = config_processing.end_date

# TODO: hashtags aus Text entfernen
stop_words = list(stopwords.words('german'))
specific_stopwords = ['ja', 'äh', 'ne']
stop_words.extend(specific_stopwords)

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
    partei_df['text'] = partei_df['video_description'].fillna('') 
    # TODO: mit oder ohne gesprochenem Text?: + " " + partei_df['voice_to_text'].fillna('')
    texts = partei_df['text'].astype(str).str.lower()
    texts = texts.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    vectorizer = CountVectorizer(
        max_df=0.7,
        min_df=5,
        max_features=2000,
        ngram_range=(1,2),
        stop_words=stop_words
    )
    X = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(n_components=6, max_iter=30, random_state=42)
    lda.fit(X)

    words = vectorizer.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[-10:][::-1]]  # 10 Top-Wörter pro Topic
        topics.append({'topic': idx+1, 'top_words': ', '.join(top_words)})

    # Speichere die Topics als CSV
    topics_df = pd.DataFrame(topics)
    topics_df.to_csv(os.path.join(results_dir, f"{partei}_{start_date}_{end_date}_lda_topics.csv"), index=False)