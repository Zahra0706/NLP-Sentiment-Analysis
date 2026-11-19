"""Interface Streamlit pour l'analyse de sentiments IMDb avec DistilBERT et XAI."""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from xai import explain_prediction


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MAX_LENGTH = 256

st.set_page_config(
    page_title="FRAME | Sentiment Intelligence",
    page_icon=":material/movie_filter:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

/* Palette unifiée. Les couleurs de texte et de fond sont fixées ensemble. */
:root {
  --page:#0B1422; --sidebar:#101D2E; --panel:#17263A;
  --panel-hover:#20334A; --border:#34465C;
  --text:#F4F7FD; --muted:#C2D0E0; --accent:#82E7C5;
  --accent-ink:#08241C; --warning:#FFD39B;
}
.stApp, [data-testid="stAppViewContainer"] {
  background: radial-gradient(ellipse 64% 44% at 80% 0%, #193446 0%, transparent 75%), var(--page);
  color:var(--text);
}
html, body, .stApp, .stApp button, .stApp textarea, .stApp input,
.stApp select {font-family:'DM Sans',sans-serif;}
.stApp h1,.stApp h2,.stApp h3,.stApp h4,
.brand,.result-title {font-family:'Space Grotesk',sans-serif;letter-spacing:-.025em;color:var(--text);}
.stApp p, .stApp li, .stApp label, .stApp small,
.stApp [data-testid="stMarkdownContainer"] {color:var(--text);}
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] p {color:var(--muted) !important;}
.stApp a {color:var(--accent);}
.stApp hr {border-color:var(--border);}

/* Sidebar : lisibilité de la navigation, slider et info. */
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
  background:var(--sidebar) !important;
}
section[data-testid="stSidebar"] {border-right:1px solid var(--border);}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {color:var(--text);}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color:var(--muted) !important;}
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
  color:var(--text) !important;border-radius:8px;padding:6px 4px;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {background:var(--panel-hover);}
section[data-testid="stSidebar"] [data-testid="stSlider"] {color:var(--text);}

/* Champ texte : pas de textarea blanche avec caractères pâles. */
.stApp [data-testid="stTextArea"] label,
.stApp [data-testid="stTextArea"] label p {color:var(--text) !important;font-weight:600;}
.stApp [data-testid="stTextArea"] textarea,
.stApp [data-testid="stTextArea"] textarea:focus {
  background:var(--panel) !important;
  color:var(--text) !important;
  -webkit-text-fill-color:var(--text) !important;
  border:1px solid var(--border) !important;
  border-radius:12px !important;
  caret-color:var(--accent) !important;
  box-shadow:none !important;
}
.stApp [data-testid="stTextArea"] textarea::placeholder {
  color:#ADC0D4 !important; -webkit-text-fill-color:#ADC0D4 !important;
  opacity:1 !important;
}
.stApp [data-testid="stTextArea"] textarea:focus {border-color:var(--accent) !important;}

/* Boutons et états hover / focus. */
.stApp .stButton > button,
.stApp .stDownloadButton > button {
  border-radius:11px !important;
  min-height:42px;
  font-weight:700;
  transition:background-color .15s ease,border-color .15s ease;
}
.stApp .stButton > button[kind="primary"] {
  background:var(--accent) !important;color:var(--accent-ink) !important;
  border:1px solid var(--accent) !important;
}
.stApp .stButton > button[kind="primary"] * {color:var(--accent-ink) !important;fill:var(--accent-ink) !important;}
.stApp .stButton > button[kind="primary"]:hover {background:#AEF5DB !important;}
.stApp .stButton > button:not([kind="primary"]),
.stApp .stDownloadButton > button {
  background:var(--panel) !important;color:var(--text) !important;
  border:1px solid var(--border) !important;
}
.stApp .stButton > button:not([kind="primary"]):hover,
.stApp .stDownloadButton > button:hover {border-color:var(--accent) !important;background:var(--panel-hover) !important;}
.stApp .stButton > button:not([kind="primary"]) *,
.stApp .stDownloadButton > button * {color:var(--text) !important;}

/* Cartes métriques et graphique. */
.stApp [data-testid="stMetric"] {
  background:var(--panel);padding:20px;border:1px solid var(--border);border-radius:14px;
}
.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricLabel"] p {color:var(--muted) !important;}
.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricValue"] * {
  color:var(--text) !important;font-family:'Space Grotesk',sans-serif;
}
.stApp [data-testid="stAlert"] {border-radius:12px;}
.stApp [data-testid="stAlert"] p,
.stApp [data-testid="stAlert"] span {color:inherit;}
.stApp [data-testid="stExpander"] {background:var(--panel);border:1px solid var(--border);border-radius:12px;}

/* Blocs éditoriaux spécifiques à FRAME. */
.brand {font-weight:700;letter-spacing:.14em;font-size:1.06rem;color:var(--accent);}
.eyebrow {text-transform:uppercase;letter-spacing:.16em;color:var(--accent);font-size:.73rem;font-weight:700;}
.subtitle {color:var(--muted) !important;font-size:1.02rem;margin-top:-8px;margin-bottom:24px;}
.result-card {
  border:1px solid var(--border);background:linear-gradient(125deg,#1B3043,#17273A);
  padding:22px 24px;border-radius:15px;margin:12px 0;
}
.result-title {font-size:1.7rem;font-weight:700;line-height:1.35;}
.result-overline {text-transform:uppercase;letter-spacing:.13em;color:var(--muted);font-size:.77rem;}
.explain {
  line-height:2.55;font-size:1.06rem;color:var(--text);
  background:var(--panel);border:1px solid var(--border);
  padding:20px;border-radius:14px;overflow-wrap:anywhere;
}
.token {padding:4px 6px;margin:2px;border-radius:5px;display:inline;box-decoration-break:clone;-webkit-box-decoration-break:clone;}
.footer {color:var(--muted);font-size:.83rem;border-top:1px solid var(--border);padding-top:16px;margin-top:36px;}

/* Amélioration de l'accessibilité au clavier. */
.stApp button:focus-visible,.stApp textarea:focus-visible {outline:2px solid var(--accent) !important;outline-offset:2px;}
@media (max-width: 750px) {
  .result-card {padding:16px;}
  .stApp [data-testid="stMetric"] {padding:13px;}
  .stApp h1 {font-size:2rem;}
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Chargement du modèle DistilBERT...")
def load_artifacts():
    """Charge une seule fois le modèle et son tokenizer depuis le disque."""
    if not (MODEL_DIR / "config.json").exists():
        raise FileNotFoundError(
            f"Dossier modèle introuvable : {MODEL_DIR}. "
            "Extraire best_distilbert dans le dossier model/."
        )
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return tokenizer, model, device


def predict(text: str, tokenizer, model, device):
    """Retourne la classe et les probabilités calculées par DistilBERT."""
    batch = tokenizer(text, return_tensors="pt", truncation=True,
                      max_length=MAX_LENGTH)
    batch = {name: tensor.to(device) for name, tensor in batch.items()}
    with torch.inference_mode():
        logits = model(**batch).logits
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    predicted_class = int(max(range(len(probabilities)), key=probabilities.__getitem__))
    labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    return {
        "class_id": predicted_class,
        "sentiment": labels.get(predicted_class, str(predicted_class)),
        "positive": probabilities[1],
        "negative": probabilities[0],
        "confidence": probabilities[predicted_class],
    }


def highlight_html(explanation: dict) -> str:
    """Colorie les mots selon leur contribution à la classe prédite."""
    pieces = []
    position = 0
    text = explanation["analyzed_text"]
    for item in explanation["words"]:
        a, b = item["start"], item["end"]
        if a < position or b > len(text):
            continue
        pieces.append(html.escape(text[position:a]))
        intensity = abs(item["relative_score"])
        opacity = 0.12 + 0.52 * intensity
        if item["score"] >= 0:
            background = f"rgba(92,218,171,{opacity:.3f})"
        else:
            background = f"rgba(242,160,110,{opacity:.3f})"
        tooltip = html.escape(f"Contribution : {item['score']:+.4f}", quote=True)
        pieces.append(
            f'<span class="token" style="background:{background}" '
            f'title="{tooltip}">{html.escape(text[a:b])}</span>'
        )
        position = b
    pieces.append(html.escape(text[position:]))
    return '<div class="explain">' + "".join(pieces).replace("\n", "<br>") + '</div>'


with st.sidebar:
    st.markdown('<div class="brand">FRAME / NLP</div>', unsafe_allow_html=True)
    st.caption("IMDb REVIEW INTELLIGENCE · V1.0")
    st.divider()
    st.markdown("**Navigation**")
    page = st.radio("Section", ["Analyse", "À propos du modèle"],
                    label_visibility="collapsed")
    st.divider()
    st.markdown("**Paramètres XAI**")
    steps = st.select_slider("Précision de l'intégration", options=[8, 16, 24, 32], value=24)
    st.caption("Plus d'étapes = calcul plus long, en particulier sur CPU.")
    st.info("Le calcul XAI est optionnel et se lance après la prédiction.", icon=":material/info:")

try:
    tokenizer, model, device = load_artifacts()
except Exception as exc:
    st.error(f"Impossible de charger le modèle : {exc}", icon=":material/error:")
    st.stop()

if page == "À propos du modèle":
    st.markdown('<div class="eyebrow">Model documentation</div>', unsafe_allow_html=True)
    st.title("À propos de DistilBERT")
    st.write("Modèle Transformer fine-tuné pour la classification binaire des critiques de films IMDb.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy (test)", "92,03 %")
    c2.metric("F1-score (test)", "92,06 %")
    c3.metric("ROC-AUC (test)", "97,50 %")
    st.caption("Valeurs issues de l'évaluation antérieure du projet, non recalculées dans cette interface.")
    st.markdown("**Pipeline** : texte anglais → tokenizer → DistilBERT → logits → softmax → sentiment.")
    st.markdown("**Explication locale** : Layer Integrated Gradients (Captum), sur les embeddings du modèle.")
    st.markdown("**Limites** : avis ironiques, ambiguës ou au-delà de 256 tokens ; scores softmax non calibrés.")
    st.markdown(f"**Exécution actuelle** : `{device.type.upper()}`")
    st.stop()

st.markdown('<div class="eyebrow">Film review intelligence / NLP</div>', unsafe_allow_html=True)
st.title("Understand the sentiment behind every review.")
st.markdown('<p class="subtitle">Analyse des critiques de films avec DistilBERT et explications locales des décisions.</p>', unsafe_allow_html=True)

left, right = st.columns([2.2, 1], gap="large")
with left:
    review = st.text_area(
        "Critique à analyser (anglais)",
        placeholder="The acting was impressive, but the ending felt rushed...",
        height=180,
        key="review_input",
    )
    analyze = st.button("Analyser le sentiment", type="primary", icon=":material/analytics:",
                        use_container_width=True)
with right:
    st.markdown("#### Aperçu du système")
    st.caption("Classification binaire")
    st.markdown("**Classes** : Positive / Negative")
    st.markdown("**Modèle** : DistilBERT fine-tuné")
    st.markdown("**XAI** : Integrated Gradients")
    st.markdown(f"**Moteur** : `{device.type.upper()}`")
    st.caption("La saisie est traitée localement par le modèle chargé.")

if analyze:
    if not review.strip():
        st.warning("Veuillez saisir une critique avant l'analyse.", icon=":material/warning:")
    elif len(review.strip()) < 3:
        st.warning("Veuillez saisir une critique plus longue.", icon=":material/warning:")
    else:
        with st.spinner("Analyse du sentiment..."):
            result = predict(review, tokenizer, model, device)
        st.session_state["prediction"] = {"text": review, "result": result}
        st.session_state.pop("explanation", None)

saved = st.session_state.get("prediction")
if saved and saved["text"] == review:
    result = saved["result"]
    st.divider()
    st.markdown('<div class="eyebrow">Prediction / Result</div>', unsafe_allow_html=True)
    sentiment = result["sentiment"].capitalize()
    st.markdown(
        f'<div class="result-card"><div class="result-overline">Sentiment détecté</div>'
        f'<div class="result-title">{html.escape(sentiment)}</div></div>',
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Positive", f"{result['positive']:.2%}")
    col2.metric("Negative", f"{result['negative']:.2%}")
    col3.metric("Score du modèle", f"{result['confidence']:.2%}")
    st.caption("Le score affiché est une probabilité softmax estimée, pas une fiabilité garantie.")

    st.markdown("### Explicabilité de la prédiction")
    st.write("Les mots verts soutiennent la classe prédite ; les mots orange s'y opposent, par rapport à une entrée de référence.")
    if st.button("Calculer l'explication XAI", icon=":material/psychology:", type="secondary"):
        with st.spinner("Calcul des contributions (Captum)..."):
            try:
                explanation = explain_prediction(
                    review, model, tokenizer, result["class_id"],
                    max_length=MAX_LENGTH, n_steps=steps,
                )
                st.session_state["explanation"] = {
                    "text": review,
                    "class_id": result["class_id"],
                    "steps": steps,
                    "data": explanation,
                }
            except Exception as exc:
                st.error(f"Le calcul XAI a échoué : {exc}", icon=":material/error:")

    stored = st.session_state.get("explanation")
    if (stored and stored["text"] == review
            and stored["class_id"] == result["class_id"]
            and stored["steps"] == steps):
        explanation = stored["data"]
        st.markdown(highlight_html(explanation), unsafe_allow_html=True)
        st.caption("Vert : soutient la décision. Orange : s'oppose à la décision. L'intensité représente l'amplitude relative de la contribution.")
        if explanation["truncated"]:
            st.warning("Seule la portion initiale du texte (256 tokens au maximum) a été analysée.")

        rows = sorted(explanation["words"], key=lambda x: abs(x["score"]), reverse=True)[:12]
        if rows:
            st.markdown("**Mots les plus influents**")
            chart = pd.DataFrame({
                "Mot": [item["word"] for item in rows],
                "Contribution": [item["score"] for item in rows],
            }).set_index("Mot")
            st.bar_chart(chart, horizontal=True)
        st.caption(f"Écart de convergence (valeur absolue moyenne) : {explanation['convergence_delta']:.5f}")
        st.caption("L'attribution représente une approximation locale du comportement du modèle ; ce n'est pas une preuve causale.")

    export = {
        "review": review,
        "prediction": result,
        "xai": st.session_state.get("explanation", {}).get("data")
               if st.session_state.get("explanation", {}).get("text") == review else None,
    }
    st.download_button(
        "Exporter l'analyse (JSON)",
        data=json.dumps(export, indent=2, ensure_ascii=False),
        file_name="imdb_sentiment_analysis.json",
        mime="application/json",
        icon=":material/download:",
    )

st.markdown('<p class="footer">FRAME NLP · Projet universitaire — IMDb Reviews · DistilBERT + Captum</p>', unsafe_allow_html=True)
