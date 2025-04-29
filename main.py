# ========== Imports ==========

import os
import tempfile
import webbrowser

import faiss
import google.generativeai as genai
import ipywidgets as widgets
import numpy as np
import pandas as pd
import textstat
from dotenv import load_dotenv
from elevenlabs import ElevenLabs, VoiceSettings
from IPython.display import Audio, display
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from feedback_gui import feedback_window
from nlp_utils import analyze_text

# ========== Chargement des variables d'environnement ==========

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

model = genai.GenerativeModel("gemini-2.0-flash")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ========== Chargement des données ==========


def load_and_process_data():
    df_social = pd.read_csv("data/Social_Media_Advertising.csv")
    df_amazon = pd.read_csv(
        "data/train.csv", sep=",", quoting=3, on_bad_lines="skip", low_memory=False
    )
    df = pd.read_csv("data/digital_marketing_campaigns_smes.csv")

    entries = []

    for _, row in df_social.iterrows():
        entries.append(
            {
                "produit": str(row.get("Campaign_Goal", "")).lower(),
                "cible": str(row.get("Target_Audience", "")).lower(),
                "format": str(row.get("Channel_Used", "")).lower(),
                "completion": f"Campagne '{row.get('Campaign_Goal', '')}' sur {row.get('Channel_Used', '')} pour {row.get('Target_Audience', '')} | ROI : {row.get('ROI', '')}, engagement : {row.get('Engagement_Score', '')}",
            }
        )

    for _, row in df_amazon.iterrows():
        bullet_points = str(row.get("BULLET_POINTS", "")).replace("\n", " ")
        description = str(row.get("DESCRIPTION", "")).replace("\n", " ")
        entries.append(
            {
                "produit": str(row.get("TITLE", "")).lower(),
                "cible": "général",
                "format": "amazon",
                "completion": f"{bullet_points} {description}",
            }
        )

    for _, row in df.iterrows():
        entries.append(
            {
                "produit": str(row.get("industry", "")).lower(),
                "cible": str(row.get("target_audience", "")).lower(),
                "format": str(row.get("marketing_channel", "")).lower(),
                "completion": f"Conversion : {row.get('conversion_rate', '')}, engagement : {row.get('engagement_rate', '')}",
            }
        )

    return entries


# ========== Construction et chargement de l'index FAISS ==========


def build_faiss_index(entries):
    texts = [f"{e['produit']} {e['cible']} {e['format']}" for e in entries]
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, "faiss_index.idx")
    np.save("embeddings.npy", embeddings)
    pd.DataFrame(entries).to_csv("entries.csv", index=False)

    return index, embeddings, texts


def load_faiss_index_and_data():
    index = faiss.read_index("faiss_index.idx")
    embeddings = np.load("embeddings.npy")
    entries = pd.read_csv("entries.csv").to_dict(orient="records")
    texts = [f"{e['produit']} {e['cible']} {e['format']}" for e in entries]
    return index, embeddings, entries, texts


def semantic_search(query, entries, embeddings, texts, index, top_k=3):
    query_vector = embed_model.encode([query])[0]
    distances, indices = index.search(np.array([query_vector]), top_k)
    return [entries[i] for i in indices[0]]


def generate_with_rag(produit, cible, format_pub, examples):
    exemples_textes = "\n".join([f"- {ex['completion']}" for ex in examples])
    prompt = (
        f"Voici des exemples de campagnes publicitaires similaires :\n"
        f"{exemples_textes}\n\n"
        f"Maintenant, génère une campagne créative adaptée au contexte suivant :\n"
        f"- Produit : {produit}\n"
        f"- Cible : {cible}\n"
        f"- Format publicitaire : {format_pub}\n"
        f"→ Le message doit être engageant, original, pertinent pour une PME, et rédigé de façon professionnelle."
    )
    response = model.generate_content(prompt)
    return response.text


# ========== Feedback utilisateur ==========


def feedback_with_stars():
    star_widget = widgets.IntSlider(
        value=0,
        min=0,
        max=5,
        step=1,
        description="Étoiles :",
        style={"description_width": "initial"},
        orientation="horizontal",
    )
    display(star_widget)

    def on_value_change(change):
        feedback = change["new"]
        print(f"Votre avis : {feedback} étoiles")
        with open("feedback.txt", "a") as f:
            f.write(f"Feedback: {feedback} étoiles\n")
        print("Merci pour votre feedback !")


# ========== Génération et lecture audio ==========


# ========== Voice Descriptions ==========

voice_descriptions = {
    "Aria": "🌞 Voix féminine lumineuse et expressive, parfaite pour des récits captivants ou des messages enthousiastes.",
    "Roger": "🎤 Voix masculine grave et posée, idéale pour des narrations sérieuses ou des messages institutionnels.",
    "Sarah": "💬 Voix féminine douce et rassurante, parfaite pour les messages de bienvenue et les instructions calmes.",
    "Laura": "🎵 Voix féminine chaleureuse et enthousiaste, qui transmet de l’énergie et de la bonne humeur.",
    "Charlie": "🎧 Voix masculine jeune et naturelle, idéale pour les contenus modernes et dynamiques.",
    "George": "🤔 Voix masculine mature et confiante, parfaite pour des documentaires et des discours inspirants.",
    "Callum": "🎤 Voix masculine claire et versatile, adaptée aux dialogues et messages explicatifs.",
    "River": "🎤 Voix androgynous fluide et apaisante, idéale pour des messages neutres et inclusifs.",
    "Liam": "🎤 Voix masculine jeune et posée, parfaite pour les podcasts et récits décontractés.",
    "Charlotte": "🎶 Voix féminine douce et classique, adaptée aux narrations littéraires et aux contes.",
    "Alice": "🎧 Voix féminine moderne et pétillante, parfaite pour les contenus lifestyle et les tutoriels.",
    "Matilda": "🌸 Voix féminine douce et réconfortante, idéale pour les livres audio ou les méditations guidées.",
    "Will": "🎤 Voix masculine dynamique et engageante, idéale pour les vidéos promotionnelles et les pubs.",
    "Jessica": "🎤 Voix féminine chaleureuse et souriante, parfaite pour les messages d’accueil et les vidéos sociales.",
    "Eric": "🎙️ Voix masculine sérieuse et charismatique, idéale pour les discours solennels et les narrations historiques.",
    "Chris": "🎤 Voix masculine décontractée et naturelle, adaptée aux podcasts et interviews détendues.",
    "Brian": "🎧 Voix masculine claire et assurée, idéale pour les annonces et les présentations professionnelles.",
    "Daniel": "🎙️ Voix masculine posée et expressive, parfaite pour les reportages et les vidéos explicatives.",
    "Lily": "🎶 Voix féminine douce et mélodieuse, adaptée aux contes et messages poétiques.",
    "Bill": "🎤 Voix masculine énergique et entraînante, idéale pour les publicités et annonces événementielles.",
}

# ========== Génération et lecture audio ==========


def generate_audio_from_text(text, voice_id, style):
    try:
        audio_gen = client.generate(
            text=text,
            voice=voice_id,
            model="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.4, similarity_boost=0.75, style=style
            ),
        )
        if not audio_gen:
            raise ValueError("Aucune donnée audio générée.")
        audio_bytes = b"".join(audio_gen)
        if len(audio_bytes) == 0:
            raise ValueError("Les données audio sont vides.")
        return audio_bytes
    except Exception as e:
        print(f"Erreur lors de la génération de l'audio : {e}")
        return None


def play_audio(audio_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(audio_bytes)
            temp_audio_path = f.name
        print(f"Audio sauvegardé temporairement à : {temp_audio_path}")
        webbrowser.open(f"file://{temp_audio_path}")
    except Exception as e:
        print(f"Erreur lors de la lecture de l'audio : {e}")


def select_voice():
    print("\n🎤 Sélectionnez une voix pour la lecture audio :")
    for idx, (voice_name, description) in enumerate(voice_descriptions.items(), 1):
        print(f"{idx}. {voice_name}: {description}")

    choice = int(input("\nEntrez le numéro de la voix souhaitée: "))
    voice_name = list(voice_descriptions.keys())[choice - 1]
    return voice_name


# ========== Fonction principale ==========


def main():
    print("🧠 Générateur Publicitaire Intelligent\n")
    produit = input("Produit : ")
    cible = input("Cible : ")
    format_pub = input("Format : ")

    if (
        os.path.exists("faiss_index.idx")
        and os.path.exists("embeddings.npy")
        and os.path.exists("entries.csv")
    ):
        print("📂 Chargement de l'index et des données...")
        index, embeddings, entries, texts = load_faiss_index_and_data()
    else:
        print("🛠️ Construction de l'index et des données...")
        entries = load_and_process_data()
        index, embeddings, texts = build_faiss_index(entries)

    examples = semantic_search(
        f"{produit} {cible} {format_pub}", entries, embeddings, texts, index
    )
    result = generate_with_rag(produit, cible, format_pub, examples)
    print("\n✅ Résultat :\n", result)

    print(
        "\n📣 Veuillez donner votre avis sur cette campagne via l'interface graphique."
    )
    feedback_window()

    print("\n📚 Niveau de lisibilité :", textstat.flesch_reading_ease(result))
    print("🧒 Âge de compréhension :", textstat.text_standard(result))

    voice_name = select_voice()
    style = 0.5

    audio_bytes = generate_audio_from_text(result, voice_name, style)
    if audio_bytes:
        play_audio(audio_bytes)
    else:
        print("Aucun audio généré.")


# ========== Lancement du programme ==========

if __name__ == "__main__":
    main()
