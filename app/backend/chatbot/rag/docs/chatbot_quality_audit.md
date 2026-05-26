# Audit qualite conversationnelle du chatbot MASI

## Pipeline audite

Le pipeline chatbot est compose de quatre blocs :

- `chat_service.py` valide la question, nettoie l'historique, classe l'intention, construit le prompt et valide la reponse.
- `fast_markdown_retriever.py` recupere les passages RAG Markdown avec score, source, section et type de document.
- `routed_context_builder.py` selectionne le contexte chiffre et documentaire utile selon l'intention.
- `local_ollama_client.py` envoie le prompt final a Ollama avec temperature faible pour limiter les hallucinations.

## Risques identifies

Les risques principaux etaient :

- contexte RAG parfois trop large ou peu pertinent ;
- absence de score minimum avant injection RAG ;
- contexte dynamique lisible mais pas assez standardise pour le modele ;
- historique conversationnel trop long pouvant polluer les reponses ;
- absence de validation systematique des chiffres generes ;
- intentions comme "page de forecasting" laissees au modele generatif, ce qui favorisait l'invention de valeurs relatives.

## Corrections appliquees

- Ajout d'un routeur d'intention : explanation, forecast_query, risk_query, backtest_query, strategy_query, data_query, out_of_scope.
- Contexte dynamique reduit selon l'intention.
- RAG filtre par score minimum, limite a 3-5 chunks, dedoublonne et reformate avec metadata.
- Prompt final restructure en blocs SYSTEM, DYNAMIC CONTEXT, RAG CONTEXT, USER QUESTION et RESPONSE RULES.
- Validation anti-hallucination avant affichage : chiffres non supportes, conseil financier, confusion VaR/ES, confusion forecast/realise et certitude excessive.
- Parametres Ollama orientes factualite : temperature faible, top_p modere, contexte configurable.

## Exemple avant / apres

Question :

> la page de forcasting

Mauvaise reponse avant :

> Rendement prevu pour aujourd'hui : -0.56%. Rendement prevu pour demain : -0.63%. Rendement prevu pour la semaine prochaine : -0.72%.

Probleme : les dates relatives et les valeurs ne viennent pas du contexte dynamique.

Reponse attendue apres :

> Horizon 1 jour, cible 2026-05-04 : rendement prevu -0.801%, VaR 5% -1.559%, ES 5% -2.274%.

La reponse doit utiliser les dates cibles exactes du dashboard.

## Exemple de refus prudent

Question :

> Est-ce que je dois acheter le MASI ?

Reponse attendue :

> Je ne peux pas donner de conseil d'investissement ni te dire quoi acheter, vendre ou allouer. En revanche, je peux t'aider a lire les previsions de rendement, la VaR, l'ES, les regimes de volatilite et le backtesting du dashboard.

## Exemple VaR / ES

Question :

> C'est quoi la VaR actuelle ?

Reponse attendue :

> La VaR actuelle a horizon 1 jour, cible 2026-05-04, est -1.559%. L'ES 5% est -2.274%. La VaR est un seuil de perte conditionnel, tandis que l'ES mesure la severite moyenne au-dela de ce seuil.
