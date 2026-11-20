"""Explications locales de DistilBERT avec Layer Integrated Gradients (Captum).

Les contributions sont calculées pour la classe prédite :
  > 0 : le mot soutient cette classe par rapport à la référence [PAD].
  < 0 : le mot s'oppose à cette classe par rapport à la référence [PAD].
Elles représentent des attributions de gradient locales et non des preuves causales.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch
from captum.attr import LayerIntegratedGradients


def explain_prediction(
    text: str,
    model: Any,
    tokenizer: Any,
    predicted_class: int,
    *,
    max_length: int = 256,
    n_steps: int = 16,
) -> dict:
    """Calcule les contributions des mots d'un avis à la classe prédite.

    Les sous-tokens WordPiece sont regroupés par mot selon les positions du
    texte original. La référence conserve [CLS]/[SEP] et remplace les autres
    identifiants par le token [PAD].
    """
    if not text.strip():
        raise ValueError("Le texte à expliquer est vide.")
    if n_steps < 2:
        raise ValueError("n_steps doit être supérieur ou égal à 2.")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Un tokenizer Hugging Face Fast est nécessaire pour les offsets.")

    device = next(model.parameters()).device
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    special_mask = encoded.pop("special_tokens_mask")[0].tolist()
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Le tokenizer n'a pas de pad_token_id.")
    baselines = input_ids.clone()
    for i, special in enumerate(special_mask):
        if not special:
            baselines[0, i] = pad_id

    def forward_func(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return model(input_ids=ids, attention_mask=mask).logits

    lig = LayerIntegratedGradients(
        forward_func,
        model.get_input_embeddings(),
    )

    model.eval()
    # L'attribution exige les gradients ; ne pas placer ce bloc sous no_grad().
    with torch.enable_grad():
        attributions, delta = lig.attribute(
            inputs=input_ids,
            baselines=baselines,
            additional_forward_args=(attention_mask,),
            target=int(predicted_class),
            n_steps=n_steps,
            internal_batch_size=min(n_steps, 8),
            return_convergence_delta=True,
        )

    token_scores = attributions.sum(dim=-1)[0].detach().cpu().numpy()
    token_scores = np.nan_to_num(token_scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Utilise les limites des mots du texte source pour réunir les sous-tokens.
    words = []
    for match in re.finditer(r"\S+", text):
        begin, end = match.span()
        contribution = 0.0
        matched = False
        for (token_begin, token_end), score, is_special in zip(
            offsets, token_scores, special_mask
        ):
            if is_special or token_end <= token_begin:
                continue
            if token_begin < end and token_end > begin:
                contribution += float(score)
                matched = True
        if matched:
            words.append({
                "word": match.group(),
                "start": begin,
                "end": end,
                "score": contribution,
            })

    max_abs = max((abs(item["score"]) for item in words), default=0.0)
    for word in words:
        word["relative_score"] = word["score"] / max_abs if max_abs else 0.0

    # Seuls les premiers tokens du texte sont analysés si la critique est longue.
    visible_ends = [end for (begin, end), special in zip(offsets, special_mask)
                    if not special and end > begin]
    analyzed_end = max(visible_ends, default=0)

    positive_words = [w for w in words if w["score"] > 0]
    negative_words = [w for w in words if w["score"] < 0]
    pos_sum = float(sum(w["score"] for w in positive_words))
    neg_sum = float(sum(w["score"] for w in negative_words))
    total_subtokens = int(attention_mask.sum().item())
    special_count = int(sum(special_mask))

    return {
        "words": words,
        "convergence_delta": float(delta.detach().cpu().abs().mean().item()),
        "analyzed_text": text[:analyzed_end],
        "truncated": analyzed_end < len(text.rstrip()),
        "target_class": int(predicted_class),
        "n_steps": n_steps,
        "positive_words": sorted(positive_words, key=lambda x: x["score"], reverse=True),
        "negative_words": sorted(negative_words, key=lambda x: x["score"]),
        "positive_sum": pos_sum,
        "negative_sum": neg_sum,
        "total_subtokens": total_subtokens,
        "effective_subtokens": total_subtokens - special_count,
    }
