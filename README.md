# Bachelorarbeit-politische-Kommunikation-auf-TikTok

Dieses repository enthält den Code und die erzeugten Grafiken der Bachelorarbeit **Politische Kommunikation auf TikTok: eine vergleichende Analyse der Bundestagsparteien**.

## Inhalt
Das Projekt umfasst:
- Extraktion und Vorverarbeitung von TikTok Video- und Kommentardaten
- Themenklassifikation mit GPT-4.1-nano nach Fine-Tuning
- Stimmungsanalyse der Kommentartexte mit einem vortrainierten BERT-Modell
- Stimmungsanalyse der Emojis in den Kommentaren mit einem Emoji-Lexikon
- Auswertung und Visualisierungen der Ergebnisse, inklusive deskriptiver Analysen

## Datenzugang
Die Daten aus dem Ordner data und die Kennzahlen der Auswertungen im Ordner results sind nicht in diesem repository enthalten. Sie sind auch Sync+Share hochgeladen, der Link befindet sich im elektronischen Anhang der Arbeit.

## Projektstruktur
```
├──data                                       # Auf Sync+Share: Ordner mit allen Daten für die Auswertung
│  └── data_preprocessed                          # Ordner für die Daten nach der Vorverarbeitung
│      └── comments                               # Kommentardaten für jedes untersuchte TikTik-Profil als CSV-Dateien
│      └── party                                  # Kommentar- und Videodaten jeweils gegliedert nach Partei als CSV-Dateien
│      └── videos                                 # Videodaten für jedes untersuchte Profil als CSV-Dateien
│          └── labeled                            # Ordner mit den vorverarbeiteten gelabelten Videodaten für das Fine-Tuning als CSV-Dateien
│  └── data_raw                                   # Ordner für alle Daten vor der Vorverarbeitung
│      └── comments                               # Kommentardaten der TikTok API für jedes untersuchte TikTok Profil als CSV-Dateien
│          └── no_comments_start_end.txt          # Liste der Nutzer:innen ohne Kommentare im batch start-end
│          └── no_comments_20250101_20250223.txt  # Liste der Nutzer:innnen ohne Kommentare im gesamten Untersuchungszeitraum
│          └── ..._comments_20250101_20250223.csv # Kommentardaten pro username im Untersuchungszeitraum als CSV-Dateien
│      └── userinfo                               # Nutzerinformationen der TikTok API für jedes untersuchte TikTok Profil als CSV-Dateien
│      └── videos                                 # Videodaten der TikTok API für jedes untersuchte TikTok Profil als CSV-Dateien
│          └── labeled                            # Ordner mit den gelabelten Videodaten für das Fine-Tuning
│              └── topic_examples.jsonl           # Im Code erzeugte Datei für das Fine-Tuning (genutzt: s. topic_examples_original.jsonl)
│      └── emoji_sentiment_data.csv               # Emoji-Sentiment-Lexikon
│      └── no_data_usernames.txt                  # Liste der Nutzer:innen ohne Videodaten
│      └── topic_examples_original.jsonl          # Genutztes gelabeltes subset für das Fine-Tuning des LLM
├──plots                                      # Auf GitHub: Ordner für die im Code erzeugten Plots/Grafiken
│  └── descriptive_analysis                       # Ordner für die Grafiken der deskriptiven Analyse
│  └── sentiment_analysis                         # Ordner für die Grafiken der Stimmungsanalyse
│  └── topic_analysis                             # Ordner für die Grafiken der Themenanalyse
│      └── no_wahlkampf                           # Ordner für die Plots der Analysen ohne den Themenbereich Wahlkampf
├──results                                    # Auf Sync+Share: Ordner für die Ergebnisse, ausgenommen Grafiken
│  └── descriptive_analysis                       # Ordner für die Ergebnisse der deskriptiven Analyse in CSV-Dateien
│  └── emoji_analysis                             # Ordner für die Ergebnisse der Emoji-Analyse in CSV-Dateien
│  └── sentiment_analysis                         # Ordner für die Ergebnisse der Stimmungsanalyse in CSV-Dateien
│      └── 20250101_20250223                      # Ordner für die Ergebnisse der Sentimentanalyse im Untersuchungszeitraum in CSV-Dateien
│      └── 20250101_20250223_with_emoji_sentiment # Ordner für die Ergebnisse der Sentimentanalyse mit Emoji-Sentiment in CSV-Dateien
│  └── topic_analysis                             # Ordner für die Ergebnisse der Themenanalyse in CSV-Dateien
│      └── cleaned                                # Ordner für die Videos mit Themenzuordnung als CSV-Dateien
│          └── subset_labeled                     # Ordner für das manuell gelabelte subset der Videodaten ohne Thema als CSV-Dateien
│              └── processed                      # Verarbeitete manuell mit Themen gelabelte Videodaten als CSV-Dateien
│          └── subset_not_labeled                 # Videodaten ohne zugeordnetes Thema als CSV-Dateien, ohne labeling
│      └── engagement_metrics                     # Extrahierte Engagementsmetriken pro Partei und Themenbereich
│      └── merged                                 # Videodaten mit Themezuordnung, zusammengefügt mit allen Videoeigenschaften
│      └── metrics                                # Analyse der Beliebtheitsmetriken pro Partei (unabhängig von Themenbereich)
│      └── no_wahlkampf                           # Ordner für die Analyse ohne den Themenbereich Wahlkampf (nicht genutzt)
│      └── output_openai                          # Videodaten mit den Antworten von GPT-4.1-nano zur Themenzuordnung
├──scripts                                    # Auf GitHub: Ordner für die zur Auswertung verwendeten Skripte
│  └── analysis                                   # Skripte für die Datenanalyse
│      └── config_analysis.py                     # Konfigurationsdatei für die Analyse
│      └── sentiment_analysis.py                  # Code für die Zuordnung der Textstimmung
│      └── sentiment_emoji_analysis.py            # Code für die Stimmungsanalyse der Emojis
│      └── topic_analysis.py                      # Code für Klassifikation der Themenbereiche durch GPT-4.1-nano
│  └── data_processing                            # Skripte für die Datengewinnung und -verarbeitung
│      └── config_processing.py                   # Konfigurationsdatei für die Datenverarbeitung
│      └── get_comments.py                        # Code für das Laden der Kommentare über die TikTok-API
│      └── get_userinfo_videos.py                 # Code für das Laden der Nutzer:inneninformationen und Videodaten über die TikTok-API
│      └── preprocess_data.py                     # Code für die Datenvorverarbeitung
│      └── preprocess_labeled_data.py             # Code für die Datenvorverarbeitung der gelabelten Videodaten für das Fine-Tuning
│  └── evaluation                                 # Skripte für die Datenauswertung
│      └── config.py                              # Konfigurationsdatei für die Datenauswertung
│      └── descriptive_analytics.py               # Code für die deskriptiven Analysen
│      └── sentiment_evaluation.py                # Code für die Auswertung der Stimmungsanalyse
│      └── topic_evaluation.py                    # Code für die Auswertung der Themenklassifikation
├──requirements.txt                               # Verwendete Pakete und Versionen für die Auswertungen
```
