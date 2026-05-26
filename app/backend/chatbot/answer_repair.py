"""
Answer Repair for the MASI Risk Dashboard Chatbot.

Role:
- Fix high-confidence factual errors before any response reaches the UI.
- Operates on raw LLM output using regex / str.replace rules.
- Does NOT call the LLM; does NOT read CSV/JSON files.

Usage:
    from backend.chatbot.answer_repair import repair_known_masi_errors
    answer = repair_known_masi_errors(raw_llm_output)
"""

from __future__ import annotations

import re


def repair_known_masi_errors(answer: str) -> str:
    """Fix a set of high-confidence factual errors that must never reach the UI.

    Rules applied (in order):
    1. Strip meta-commentary lines ("D'accord, je vais répondre directement…").
    2. Replace bracket placeholders ([…]) with "indisponible".
    3. Correct HMM regime → direction misreadings.
    4. Correct "MASI = Algerian market" hallucination.
    5. Correct "model configured for a regime" phrasing.
    6. Replace Monte Carlo references for 10/25-day horizons.
    7. Fix risk-managed strategy naming errors.
    8. Fix VaR language misreadings (perte maximale, notre portefeuille…).
    """

    repaired = answer

    # --- 1. Strip meta-commentary lines -----------------------------------------
    meta_patterns = (
        r"^\s*D['' ]?accord,?\s+je vais r(?:e|Ã©)pondre directement[^.]*\.\s*",
        r"^\s*Voici une r(?:e|Ã©)ponse attendue\s*:?\s*",
        r"^\s*Nous avons actuellement les valeurs suivantes dans le contexte dynamique\s*:?\s*",
    )
    for pattern in meta_patterns:
        repaired = re.sub(pattern, "", repaired, flags=re.IGNORECASE)

    repaired = repaired.replace("dans le contexte dynamique", "dans le dashboard")
    repaired = re.sub(r"\[[^\]]+\]", "indisponible", repaired)
    repaired = repaired.replace(
        "les faits methodologiques non negociables", "la methodologie du projet"
    )
    repaired = repaired.replace(
        "les faits mÃ©thodologiques non nÃ©gociables", "la methodologie du projet"
    )
    repaired = re.sub(
        r"\n?\s*Si vous avez besoin[^.]*\.\s*$",
        "",
        repaired,
        flags=re.IGNORECASE,
    )
    repaired = re.sub(
        r"\n?\s*N['']?h(?:e|Ã©)sitez pas[^.]*\.\s*$",
        "",
        repaired,
        flags=re.IGNORECASE,
    )

    # Filter line-level meta junk
    cleaned_lines = []
    meta_needles = (
        "je vais repondre directement",
        "je vais r?pondre directement",
        "voici une reponse attendue",
        "voici une r?ponse attendue",
        "informations fournies",
        "nous avons actuellement les valeurs suivantes",
        "rendement du masi est de indisponible",
        "volatilite est de indisponible",
        "volatilité est de indisponible",
        "risque mesure",
        "indisponible%",
        "faits methodologiques",
        "faits m?thodologiques",
    )
    for line in repaired.splitlines():
        normalized_line = (
            line.lower()
            .replace("Ã©", "e")
            .replace("Ã¨", "e")
            .replace("Ãª", "e")
            .replace("é", "e")
            .replace("è", "e")
            .replace("ê", "e")
        )
        if any(needle in normalized_line for needle in meta_needles):
            continue
        cleaned_lines.append(line)
    repaired = "\n".join(cleaned_lines)

    # Strip boilerplate injected by the prompt
    boilerplate_patterns = (
        r"\s*Le r(?:e|Ã©|\?)gime HMM est estim(?:e|Ã©|\?)[^.]*\.",
        r"\s*Les horizons de 10 jours et 25 jours sont pr(?:e|Ã©|\?)sent(?:e|Ã©|\?)s comme des extensions math(?:e|Ã©|\?)matiques[^.]*\.",
        r"\s*La simulation risk-managed est une analyse bas(?:e|Ã©|\?)e sur des donn(?:e|Ã©|\?)es historiques[^.]*\.",
    )
    for pattern in boilerplate_patterns:
        repaired = re.sub(pattern, "", repaired, flags=re.IGNORECASE)
    repaired = repaired.strip()

    lower_answer = repaired.lower()

    # --- 2. HMM direction misreadings -------------------------------------------
    direction_misreadings = (
        "regime hmm est actuellement en phase de baisse",
        "régime hmm est actuellement en phase de baisse",
        "regime hmm confirme",
        "régime hmm confirme",
        "confirmee par le regime hmm",
        "confirmée par le régime hmm",
        "confirmee par le regime",
        "confirmée par le régime",
    )
    if any(pattern in lower_answer for pattern in direction_misreadings):
        repaired = re.sub(
            r"Le r(?:e|é)gime HMM est actuellement en phase de baisse\.",
            "Le regime HMM indique un etat de volatilite, pas une direction de prix.",
            repaired,
            flags=re.IGNORECASE,
        )
        repaired = re.sub(
            r"Le MASI est actuellement dans une phase de baisse,\s*confirm(?:e|é)ée? par le r(?:e|é)gime HMM\.",
            (
                "Le rendement prevu negatif suggere une baisse conditionnelle du MASI, "
                "mais le regime HMM ne confirme pas la direction : il renseigne seulement sur la volatilite."
            ),
            repaired,
            flags=re.IGNORECASE,
        )
        repaired = re.sub(
            r"confirm(?:e|é)ée? par le r(?:e|é)gime(?: HMM)?",
            "indiquee par le rendement prevu negatif, pas par le regime HMM",
            repaired,
            flags=re.IGNORECASE,
        )
        lower_answer = repaired.lower()

    hmm_direction_patterns = (
        r"[^.]*r(?:e|é|\?)gime HMM[^.]*(?:baisse|hausse|direction|confirme|classifi(?:e|é|\?))[^.]*\.",
        r"[^.]*r(?:e|é|\?)gime[^.]*(?:baisse|hausse|direction|confirme)[^.]*\.",
        r"[^.]*HMM[^.]*(?:pr(?:e|é)di(?:re|t)|anticip(?:e|er)|tendance|trend)[^.]*\.",
    )
    if any(re.search(p, repaired, flags=re.IGNORECASE) for p in hmm_direction_patterns):
        for p in hmm_direction_patterns:
            repaired = re.sub(
                p,
                " Le regime HMM indique seulement un etat de volatilite, pas une direction du MASI.",
                repaired,
                flags=re.IGNORECASE,
            )
        lower_answer = repaired.lower()

    repaired = re.sub(
        r"VaR pr(?:e|é)dite par le mod(?:e|è)le HMM",
        "VaR predite par le modele de risque",
        repaired,
        flags=re.IGNORECASE,
    )

    volatility_direction_patterns = (
        r"[^.]*baisse[^.]*en raison de la volatilit(?:e|é|\?)[^.]*\.",
        r"[^.]*baisse[^.]*avec une augmentation de la volatilit(?:e|é|\?)[^.]*\.",
    )
    if any(re.search(p, repaired, flags=re.IGNORECASE) for p in volatility_direction_patterns):
        for p in volatility_direction_patterns:
            repaired = re.sub(
                p,
                (
                    " La baisse estimee doit etre lue a partir du rendement prevu negatif; "
                    "la volatilite mesure l'incertitude autour de cette prevision."
                ),
                repaired,
                flags=re.IGNORECASE,
            )
        lower_answer = repaired.lower()

    # --- 3. Algerian market hallucination ----------------------------------------
    if "masi" in lower_answer and ("algér" in lower_answer or "alger" in lower_answer):
        repaired = repaired.replace(
            "Marché Algérien des Valeurs Mobilières", "marché actions marocain"
        )
        repaired = repaired.replace(
            "marché algérien des valeurs mobilières", "marché actions marocain"
        )
        repaired = repaired.replace("marché algérien", "marché marocain")
        repaired = repaired.replace("Marché algérien", "marché marocain")

    # --- 4. Regime configuration misreading --------------------------------------
    regime_misreadings = (
        "Le modele est configure pour fonctionner dans un regime",
        "Le modèle est configuré pour fonctionner dans un régime",
        "le modele est configure pour fonctionner dans un regime",
        "le modèle est configuré pour fonctionner dans un régime",
        "modele est configure pour fonctionner dans un regime",
        "modèle est configuré pour fonctionner dans un régime",
    )
    if any(p.lower() in lower_answer for p in regime_misreadings):
        repaired = re.sub(
            r"Le mod(?:e|è)le est configur(?:e|é) pour fonctionner dans un r(?:e|é)gime[^.]*\.",
            "Le regime affiche est le regime actuellement estime par le HMM a partir des donnees recentes.",
            repaired,
            flags=re.IGNORECASE,
        )
        repaired = re.sub(
            r"modele est configure pour fonctionner dans un regime[^.]*\.",
            "Le regime affiche est le regime actuellement estime par le HMM a partir des donnees recentes.",
            repaired,
            flags=re.IGNORECASE,
        )
        repaired = repaired.replace(
            "Les previsions actuelles pour l'horizon de 1 jour, 10 jours et 25 jours sont basees sur ce regime",
            "Les previsions actuelles pour les horizons 1 jour, 10 jours et 25 jours sont conditionnees par ce regime estime",
        )
        repaired = repaired.replace(
            "Les prévisions actuelles pour l'horizon de 1 jour, 10 jours et 25 jours sont basées sur ce régime",
            "Les previsions actuelles pour les horizons 1 jour, 10 jours et 25 jours sont conditionnees par ce regime estime",
        )

    # --- 5. Monte Carlo for 10/25-day horizons -----------------------------------
    if "monte carlo" in repaired.lower():
        repaired = re.sub(
            r"Les pr(?:e|é)visions pour les horizons de 10 jours et 25 jours sont calcul(?:e|é)es en utilisant la m(?:e|é)thode de Monte Carlo\.[\s\S]*?(?=Les valeurs actuelles|Rendement pr(?:e|é)vu|$)",
            (
                "Les previsions 10 jours et 25 jours ne sont pas calculees par Monte Carlo dans cette application. "
                "Elles sont des extensions operationnelles du modele 1 jour : la moyenne conditionnelle est agregee "
                "sur l'horizon, tandis que la volatilite, la VaR et l'ES sont etendues par un scaling en racine de l'horizon.\n\n"
            ),
            repaired,
            flags=re.IGNORECASE,
        )
        repaired = repaired.replace(
            "méthode de Monte Carlo", "extension operationnelle du modele 1 jour"
        )
        repaired = repaired.replace(
            "methode de Monte Carlo", "extension operationnelle du modele 1 jour"
        )

    # --- 6. Risk-managed strategy naming errors ----------------------------------
    economic_hallucination_markers = (
        "simulation de la strategie de rendement",
        "simulation de la stratégie de rendement",
        "simulation de la strategie de volatilite",
        "simulation de la stratégie de volatilité",
    )
    if any(m in repaired.lower() for m in economic_hallucination_markers):
        for old, new in (
            ("simulation de la strategie de rendement", "simulation risk-managed basee sur la VaR predite"),
            ("simulation de la stratÃ©gie de rendement", "simulation risk-managed basee sur la VaR predite"),
            ("simulation de la strategie de volatilite", "simulation risk-managed basee sur la VaR predite"),
            ("simulation de la stratÃ©gie de volatilitÃ©", "simulation risk-managed basee sur la VaR predite"),
        ):
            repaired = repaired.replace(old, new)

    weight_return_misreadings = (
        "Le poids dans la stratégie de gestion de risque dépend de la prévision de rendement",
        "Le poids dans la strategie de gestion de risque depend de la prevision de rendement",
        "Le poids risk-managed depend de la prevision de rendement",
        "Le poids risk-managed dépend de la prévision de rendement",
    )
    if any(p.lower() in repaired.lower() for p in weight_return_misreadings):
        for p in weight_return_misreadings:
            repaired = repaired.replace(p, "Le poids risk-managed depend de la VaR predite")

    # --- 7. VaR language misreadings ---------------------------------------------
    risk_language_replacements = (
        (
            r"la VaR à 5% suggère que nous devrions prévoir une perte de ([^\.]+)\.",
            r"la VaR à 5% indique un seuil conditionnel de perte de \1.",
        ),
        (
            r"la VaR a 5% suggere que nous devrions prevoir une perte de ([^\.]+)\.",
            r"la VaR a 5% indique un seuil conditionnel de perte de \1.",
        ),
        (
            r"nous devrions observer environ 5% de violations de VaR",
            "le taux attendu de violations VaR est d'environ 5%",
        ),
        (r"notre prévision de VaR", "la prevision de VaR"),
        (r"notre prevision de VaR", "la prevision de VaR"),
        (r"notre portefeuille", "le MASI"),
        (r"gestionnaire de portefeuille à risque non managé", "Buy and Hold"),
        (r"gestionnaire de portefeuille a risque non manage", "Buy and Hold"),
        (r"gestionnaire de risque managé", "simulation risk-managed"),
        (r"gestionnaire de risque manage", "simulation risk-managed"),
        (r"risque managé de 5% de perte", "risque de queue mesure par la VaR 5%"),
        (r"risque manage de 5% de perte", "risque de queue mesure par la VaR 5%"),
    )
    for pattern, replacement in risk_language_replacements:
        repaired = re.sub(pattern, replacement, repaired, flags=re.IGNORECASE)

    return repaired
