"""Interface Streamlit épurée pour l'analyse de sentiments IMDb avec DistilBERT et XAI (Captum)."""
from __future__ import annotations

import html
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from xai import explain_prediction


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MAX_LENGTH = 256
XAI_STEPS = 16

# Configuration de page sobre
st.set_page_config(
    page_title="FRAME | Analyse de Sentiment & XAI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style CSS épuré, sobre et professionnel
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --bg-page: #0B1320;
  --bg-sidebar: #0E1726;
  --bg-card: #142234;
  --bg-card-hover: #192A40;
  --border-color: #23354C;
  
  --text-main: #F1F5F9;
  --text-muted: #94A3B8;
  --text-dim: #64748B;
  
  --color-pos: #10B981;
  --color-neg: #F43F5E;
  --color-accent: #38BDF8;
}

.stApp, [data-testid="stAppViewContainer"] {
  background: var(--bg-page);
  color: var(--text-main);
}

html, body, .stApp, .stApp button, .stApp textarea, .stApp input, .stApp select {
  font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp h1, .stApp h2, .stApp h3, .stApp h4, .brand-title, .verdict-title {
  font-family: 'Space Grotesk', sans-serif;
  letter-spacing: -0.02em;
  color: var(--text-main);
}

.stApp p, .stApp li, .stApp label, .stApp [data-testid="stMarkdownContainer"] {
  color: var(--text-main);
}

.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] p {
  color: var(--text-muted) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: var(--bg-sidebar) !important;
  border-right: 1px solid var(--border-color);
}

section[data-testid="stSidebar"] .block-container {
  padding-top: 1.5rem;
  padding-bottom: 2rem;
}

.sidebar-brand {
  font-size: 1.15rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  color: #FFFFFF;
  margin-bottom: 2px;
}

.sidebar-sub {
  font-size: 0.78rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 12px;
}

.sidebar-status {
  font-size: 0.78rem;
  color: #34D399;
  background: rgba(16, 185, 129, 0.1);
  border: 1px solid rgba(16, 185, 129, 0.25);
  padding: 4px 8px;
  border-radius: 6px;
  display: inline-block;
  margin-bottom: 16px;
}

.sidebar-metric {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 8px 10px;
  margin-bottom: 8px;
}

.sidebar-metric-label {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
}

.sidebar-metric-value {
  font-size: 1rem;
  font-weight: 700;
  color: var(--text-main);
  font-family: 'Space Grotesk', sans-serif;
}

.sidebar-metric-sub {
  font-size: 0.7rem;
  color: var(--text-dim);
}

/* Zone de saisie */
.stApp [data-testid="stTextArea"] textarea {
  background: var(--bg-card) !important;
  color: var(--text-main) !important;
  border: 1px solid var(--border-color) !important;
  border-radius: 8px !important;
  caret-color: var(--color-accent) !important;
  font-size: 0.95rem !important;
}

.stApp [data-testid="stTextArea"] textarea:focus {
  border-color: var(--color-accent) !important;
  box-shadow: none !important;
}

/* Boutons */
.stApp .stButton > button {
  border-radius: 8px !important;
  min-height: 40px;
  font-weight: 600;
  font-size: 0.92rem;
}

.stApp .stButton > button[kind="primary"] {
  background: #10B981 !important;
  color: #032217 !important;
  border: 1px solid #10B981 !important;
}

.stApp .stButton > button[kind="primary"]:hover {
  background: #34D399 !important;
  border-color: #34D399 !important;
}

.stApp .stButton > button:not([kind="primary"]),
.stApp .stDownloadButton > button {
  background: var(--bg-card) !important;
  color: var(--text-main) !important;
  border: 1px solid var(--border-color) !important;
}

.stApp .stButton > button:not([kind="primary"]):hover,
.stApp .stDownloadButton > button:hover {
  background: var(--bg-card-hover) !important;
  border-color: var(--color-accent) !important;
}

/* Cartes de résultat */
.verdict-box {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 20px 24px;
  margin: 14px 0 18px 0;
}

.verdict-box-pos {
  border-left: 5px solid var(--color-pos);
}

.verdict-box-neg {
  border-left: 5px solid var(--color-neg);
}

.verdict-title {
  font-size: 1.8rem;
  font-weight: 700;
  margin: 2px 0 4px 0;
}

/* Barre de distribution */
.dist-bar {
  background: #0B1320;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  height: 14px;
  width: 100%;
  display: flex;
  overflow: hidden;
  margin: 10px 0 4px 0;
}

.dist-pos {
  background: var(--color-pos);
  height: 100%;
}

.dist-neg {
  background: var(--color-neg);
  height: 100%;
}

/* Surlignage XAI */
.xai-container {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 10px;
  padding: 18px 20px;
  line-height: 2.3;
  font-size: 1rem;
  color: var(--text-main);
  overflow-wrap: anywhere;
  margin: 10px 0 16px 0;
}

.xai-token {
  display: inline-block;
  padding: 1px 6px;
  margin: 2px 2px;
  border-radius: 4px;
  cursor: help;
  font-weight: 500;
}

.footer-text {
  color: var(--text-dim);
  font-size: 0.8rem;
  border-top: 1px solid var(--border-color);
  padding-top: 14px;
  margin-top: 36px;
  text-align: center;
}
</style>
""", unsafe_allow_html=True)


# Chargement unique du modèle et du tokenizer
@st.cache_resource(show_spinner="Chargement du modèle DistilBERT...")
def load_artifacts():
    if not (MODEL_DIR / "config.json").exists():
        raise FileNotFoundError(
            f"Dossier modèle introuvable : {MODEL_DIR}."
        )
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return tokenizer, model, device


def predict(text: str, tokenizer, model, device) -> dict[str, Any]:
    batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    batch = {name: tensor.to(device) for name, tensor in batch.items()}
    with torch.inference_mode():
        logits = model(**batch).logits
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    
    predicted_class = int(max(range(len(probabilities)), key=probabilities.__getitem__))
    labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    sentiment = labels.get(predicted_class, "positive" if predicted_class == 1 else "negative")
    
    return {
        "class_id": predicted_class,
        "sentiment": sentiment,
        "positive": probabilities[1],
        "negative": probabilities[0],
        "confidence": probabilities[predicted_class],
    }


def compute_text_metrics(text: str, tokenizer) -> dict[str, Any]:
    clean = text.strip()
    words = re.findall(r"\b\w+\b", clean.lower())
    word_count = len(words)
    char_count = len(clean)
    unique_words = len(set(words))
    lexical_diversity = (unique_words / word_count * 100) if word_count > 0 else 0.0
    
    encoded = tokenizer(clean, truncation=False)
    token_count = len(encoded["input_ids"])
    token_ratio = min(1.0, token_count / MAX_LENGTH) if MAX_LENGTH > 0 else 0.0
    is_truncated = token_count > MAX_LENGTH
    
    return {
        "word_count": word_count,
        "char_count": char_count,
        "unique_words": unique_words,
        "lexical_diversity": lexical_diversity,
        "token_count": token_count,
        "token_ratio": token_ratio,
        "is_truncated": is_truncated,
    }


def highlight_html(explanation: dict) -> str:
    pieces = []
    position = 0
    text = explanation["analyzed_text"]
    
    for item in explanation["words"]:
        a, b = item["start"], item["end"]
        if a < position or b > len(text):
            continue
        pieces.append(html.escape(text[position:a]))
        intensity = abs(item["relative_score"])
        opacity = 0.18 + 0.60 * intensity
        
        if item["score"] >= 0:
            bg_color = f"rgba(16, 185, 129, {opacity:.3f})"
            border_col = "rgba(16, 185, 129, 0.4)"
            direction = "Soutient la décision"
        else:
            bg_color = f"rgba(244, 63, 94, {opacity:.3f})"
            border_col = "rgba(244, 63, 94, 0.4)"
            direction = "S'oppose à la décision"
            
        tooltip = f"{item['word']} | Contribution : {item['score']:+.4f} ({direction})"
        tooltip_attr = html.escape(tooltip, quote=True)
        word_escaped = html.escape(text[a:b])
        
        pieces.append(
            f'<span class="xai-token" style="background:{bg_color}; border: 1px solid {border_col};" '
            f'title="{tooltip_attr}">{word_escaped}</span>'
        )
        position = b
        
    pieces.append(html.escape(text[position:]))
    return '<div class="xai-container">' + "".join(pieces).replace("\n", "<br>") + '</div>'


def render_attribution_chart(explanation: dict, top_n: int = 10):
    words = explanation.get("words", [])
    if not words:
        return
        
    sorted_words = sorted(words, key=lambda x: abs(x["score"]), reverse=True)[:top_n]
    
    data = []
    for item in sorted_words:
        score = item["score"]
        data.append({
            "Mot": item["word"],
            "Contribution": score,
            "Orientation": "Soutient" if score >= 0 else "S'oppose",
            "Couleur": "#10B981" if score >= 0 else "#F43F5E",
        })
        
    df = pd.DataFrame(data)
    
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("Contribution:Q", title="Score de contribution (Layer Integrated Gradients)"),
            y=alt.Y("Mot:N", sort=alt.EncodingSortField(field="Contribution", op="max", order="descending"), title=None),
            color=alt.Color("Couleur:N", scale=None),
            tooltip=["Mot:N", "Contribution:Q", "Orientation:N"],
        )
        .properties(height=max(180, top_n * 24))
        .configure_axis(
            gridColor="#1C2D42",
            labelColor="#A0AEC0",
            titleColor="#A0AEC0",
            labelFont="DM Sans",
            titleFont="DM Sans",
        )
        .configure_view(strokeOpacity=0)
    )
    st.altair_chart(chart, width="stretch")


# Exemples sobres de test
EXAMPLES = {
    "Avis positif classique": (
        "An absolute masterpiece of modern cinema. The storytelling is gripping, the cinematography is stunning, "
        "and the lead actors deliver an unforgettable, deeply moving performance."
    ),
    "Avis négatif classique": (
        "A colossal waste of time and money. The script is horribly clichéd, the pacing is painfully slow, "
        "and the acting is so wooden and lifeless that it becomes completely unwatchable."
    ),
    "Avis mitigé / contrasté": (
        "The visual effects and musical score were truly magnificent, but the dialogue felt clichéd and the ending "
        "was rushed and left too many plot holes unresolved."
    ),
    "Avis ironique": (
        "A truly revolutionary accomplishment if the director's ultimate goal was to cure severe insomnia. "
        "I was completely captivated by how utterly meaningless every single scene managed to be."
    ),
}

# Initialisation de l'état de session lié directement au widget de saisie
if "user_review" not in st.session_state:
    st.session_state["user_review"] = ""
if "current_analysis" not in st.session_state:
    st.session_state["current_analysis"] = None

# Chargement du modèle
try:
    tokenizer, model, device = load_artifacts()
except Exception as exc:
    st.error(f"Erreur de chargement du modèle : {exc}")
    st.stop()


# ==============================================================================
# SIDEBAR : INDICATEURS ÉPURÉS & CONTRÔLES
# ==============================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-brand">FRAME NLP</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Sentiment & XAI · IMDb</div>', unsafe_allow_html=True)
    
    device_label = "CUDA (GPU)" if device.type == "cuda" else "CPU"
    st.markdown(f'<div class="sidebar-status">Modèle opérationnel · {device_label}</div>', unsafe_allow_html=True)
    
    # 1. Indicateurs du texte en direct
    st.markdown("**Indicateurs du texte**")
    active_text = st.session_state.get("user_review", "")
    metrics = compute_text_metrics(active_text, tokenizer)
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f"""
        <div class="sidebar-metric">
            <div class="sidebar-metric-label">Mots</div>
            <div class="sidebar-metric-value">{metrics['word_count']}</div>
            <div class="sidebar-metric-sub">{metrics['char_count']} car.</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s2:
        st.markdown(f"""
        <div class="sidebar-metric">
            <div class="sidebar-metric-label">Vocabulaire</div>
            <div class="sidebar-metric-value">{metrics['lexical_diversity']:.1f}%</div>
            <div class="sidebar-metric-sub">mots uniques</div>
        </div>
        """, unsafe_allow_html=True)
        
    token_val = metrics["token_count"]
    token_pct = min(100.0, (token_val / MAX_LENGTH) * 100)
    bar_col = "#10B981" if token_val <= MAX_LENGTH else "#F43F5E"
    
    st.markdown(f"""
    <div class="sidebar-metric">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div class="sidebar-metric-label">Tokens</div>
            <div style="font-size:0.75rem; font-weight:600; color:{bar_col};">{token_val} / {MAX_LENGTH}</div>
        </div>
        <div style="background:#0B1320; border-radius:4px; height:6px; width:100%; margin-top:5px; overflow:hidden;">
            <div style="background:{bar_col}; height:100%; width:{token_pct}%;"></div>
        </div>
        <div class="sidebar-metric-sub" style="margin-top:3px;">
            {'Troncature active (>256)' if metrics['is_truncated'] else 'Dans la limite de contexte'}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # 2. Exemples prêts à charger
    st.markdown("**Exemples de test**")
    selected_preset = st.selectbox(
        "Exemples",
        options=list(EXAMPLES.keys()),
        label_visibility="collapsed",
    )
    if st.button("Charger cet exemple", width="stretch"):
        st.session_state["user_review"] = EXAMPLES[selected_preset]
        st.session_state["current_analysis"] = None
        st.rerun()
        
    st.divider()
    
    # 3. Performances du modèle
    st.markdown("**Performances du modèle (IMDb)**")
    p1, p2 = st.columns(2)
    p1.metric("Accuracy", "92.0%")
    p2.metric("F1-Score", "92.1%")
    
    st.divider()
    
    # 4. Paramètres d'affichage
    st.markdown("**Affichage**")
    top_words = st.slider("Mots dans le graphique", min_value=6, max_value=16, value=10, step=2)


# ==============================================================================
# SECTION PRINCIPALE
# ==============================================================================
st.title("Analyse de Sentiment & Explicabilité (XAI)")
st.caption("Classification binaire IMDb (DistilBERT) avec explications locales instantanées (Integrated Gradients).")

review_input = st.text_area(
    "Commentaire à analyser (en anglais) :",
    key="user_review",
    placeholder="Saisissez ou chargez un commentaire...",
    height=140,
)

c_btn1, c_btn2 = st.columns([3, 1])
with c_btn1:
    analyze = st.button("Analyser le sentiment", type="primary", width="stretch")
with c_btn2:
    if st.button("Effacer", width="stretch"):
        st.session_state["user_review"] = ""
        st.session_state["current_analysis"] = None
        st.rerun()

# Calcul automatique unique (Prediction + XAI en 1 seul clic)
if analyze:
    clean_text = review_input.strip()
    if not clean_text:
        st.warning("Veuillez saisir un commentaire avant de lancer l'analyse.")
    elif len(clean_text) < 3:
        st.warning("Le commentaire saisi est trop court.")
    else:
        with st.spinner("Analyse du sentiment et calcul de l'explication XAI..."):
            t0 = time.perf_counter()
            pred = predict(clean_text, tokenizer, model, device)
            t_infer = time.perf_counter() - t0
            
            t1 = time.perf_counter()
            xai_data = explain_prediction(
                clean_text, model, tokenizer, pred["class_id"],
                max_length=MAX_LENGTH, n_steps=XAI_STEPS
            )
            t_xai = time.perf_counter() - t1
            
            st.session_state["current_analysis"] = {
                "text": clean_text,
                "prediction": pred,
                "xai": xai_data,
                "t_infer": t_infer,
                "t_xai": t_xai,
                "total_time": t_infer + t_xai,
                "steps": XAI_STEPS,
                "timestamp": datetime.now().isoformat(),
            }

# Affichage des résultats
current = st.session_state.get("current_analysis")
if current and current["text"] == review_input.strip():
    pred = current["prediction"]
    xai_data = current["xai"]
    is_positive = pred["class_id"] == 1
    
    verdict_label = "POSITIF" if is_positive else "NÉGATIF"
    verdict_color = "#10B981" if is_positive else "#F43F5E"
    box_class = "verdict-box verdict-box-pos" if is_positive else "verdict-box verdict-box-neg"
    
    # 1. Carte de résultat
    st.markdown(f"""
    <div class="{box_class}">
        <div style="display:flex; justify-content:space-between; align-items:baseline; flex-wrap:wrap;">
            <div>
                <div style="font-size:0.8rem; text-transform:uppercase; letter-spacing:0.06em; color:#94A3B8;">Résultat détecté</div>
                <div class="verdict-title" style="color:{verdict_color};">{verdict_label}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:0.8rem; color:#94A3B8;">Confiance du modèle</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.3rem; font-weight:700; color:#FFFFFF;">
                    {pred['confidence']:.2%}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Explication locale XAI
    st.subheader("Explicabilité de la décision")
    st.caption("Vert : terme soutenant la classe prédite. Rouge : terme s'opposant à la classe prédite. Le survol d'un mot affiche sa contribution.")
    
    if xai_data:
        st.markdown(highlight_html(xai_data), unsafe_allow_html=True)
        
        if xai_data.get("truncated"):
            st.warning("Seuls les 256 premiers tokens ont été analysés par le modèle.")
            
        st.markdown(f"**Mots les plus influents (Top {top_words})**")
        render_attribution_chart(xai_data, top_n=top_words)
        
    # 3. Section détails & export (sobre)
    with st.expander("Données détaillées et export JSON", expanded=False):
        c_exp1, c_exp2 = st.columns([2, 1])
        with c_exp1:
            st.markdown("**Validation de convergence**")
            delta = xai_data.get("convergence_delta", 0.0) if xai_data else 0.0
            st.write(f"- Écart de convergence Captum (delta) : `{delta:.6f}`")
            st.write(f"- Étapes d'intégration : `{current['steps']}`")
            st.write("- Référence neutre : tokens `[PAD]`")
        with c_exp2:
            st.markdown("**Exportation**")
            export_dict = {
                "input": current["text"],
                "prediction": pred,
                "xai": xai_data,
                "duration_sec": current["total_time"],
                "timestamp": current["timestamp"],
            }
            st.download_button(
                "Télécharger le rapport JSON",
                data=json.dumps(export_dict, indent=2, ensure_ascii=False),
                file_name="analyse_sentiment_xai.json",
                mime="application/json",
                width="stretch",
            )
            
        if xai_data and xai_data.get("words"):
            st.markdown("**Tableau complet des contributions**")
            table_data = [
                {
                    "Mot": w["word"],
                    "Contribution": round(w["score"], 5),
                    "Impact relatif (%)": round(abs(w["relative_score"]) * 100, 2),
                    "Sens": "En faveur" if w["score"] >= 0 else "En opposition",
                }
                for w in xai_data["words"]
            ]
            st.dataframe(
                pd.DataFrame(table_data).sort_values("Impact relatif (%)", ascending=False),
                width="stretch",
                height=250,
            )

st.markdown('<div class="footer-text">FRAME · DistilBERT fine-tuné sur IMDb · Captum Layer Integrated Gradients</div>', unsafe_allow_html=True)
