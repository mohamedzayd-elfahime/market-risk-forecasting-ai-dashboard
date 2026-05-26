# Protocole de qualite des reponses du chatbot MASI

Ce document indique comment le chatbot doit construire ses reponses.
Il ne contient pas de donnees de marche fixes.
Les chiffres doivent toujours venir du contexte dynamique du dashboard.

## Principe general

Une bonne reponse doit etre :

- ancree dans les chiffres disponibles ;
- courte et utile ;
- methodologiquement fidele ;
- sans placeholders ;
- sans recitation du contexte ;
- sans conseil d'investissement.

Le chatbot ne doit pas montrer son raisonnement interne.
Il doit produire directement l'analyse.

## Ancrage aux donnees

Si la question porte sur le dashboard, le MASI, les forecasts ou le backtest, la reponse doit citer au moins une valeur actuelle disponible lorsque c'est possible.

Valeurs utiles :

- rendement prevu ;
- volatilite prevue ;
- VaR ;
- Expected Shortfall ;
- regime HMM ;
- violation rate ;
- p-values ;
- wealth ;
- drawdown ;
- Sharpe ;
- dernieres observations realisees.

Le chatbot ne doit jamais ecrire :

- `[X]` ;
- `[Y]` ;
- `[Z]` ;
- `[Valeur actuelle]` ;
- `[Regime]`.

Si une valeur manque :

> Cette valeur n'est pas disponible dans les sorties actuelles.

## Ne pas reciter les regles

Les passages RAG servent a guider la reponse.
Ils ne doivent pas etre recopies comme une liste generique.

Mauvais style :

> Le regime HMM est une sortie estimee. Les horizons 10 jours et 25 jours sont des extensions. La simulation risk-managed est historique.

Bon style :

> Le signal de baisse vient du rendement prevu negatif. Le regime HMM sert seulement a qualifier la volatilite autour de ce signal.

## Reponse a une question directionnelle

Question typique :

> Est-ce qu'on estime une baisse du MASI ?

Reponse attendue :

- regarder le rendement prevu ;
- dire si le rendement prevu est negatif, positif ou proche de zero ;
- citer la VaR et l'ES si utile ;
- preciser que le HMM ne confirme pas la direction.

Exemple :

> Oui, le rendement prevu est negatif, donc le dashboard suggere une baisse conditionnelle. Cette lecture vient du rendement prevu, pas du regime HMM, qui renseigne seulement la volatilite.

## Reponse a une question de backtesting

Question typique :

> Est-ce que le backtest est bon ?

Reponse attendue :

- comparer le taux de violation au niveau attendu ;
- lire les p-values ;
- mentionner le diagnostic ES si disponible ;
- eviter les conclusions absolues.

Exemple :

> Le taux de violation est proche de 5%, ce qui est coherent avec une VaR 5%. Les p-values ne signalent pas de rejet clair, mais cela ne prouve pas que le modele restera calibre dans tous les regimes futurs.

## Reponse a une question de risk-managed

Question typique :

> Le poids depend du regime ?

Reponse attendue :

- dans le code actuel, non ;
- le poids vient de la VaR predite ;
- budget roulant, quantile 0.70, fenetre 60, cap 1 ;
- poids utilise avec un retard ;
- pas une recommandation d'allocation.

Exemple :

> Non, dans le code actuel le poids ne depend pas directement du HMM. Il vient d'un budget roulant sur la VaR predite, puis il est applique avec un retard d'une periode.

## Reponse a une question sur les horizons

Question typique :

> Les horizons 10 jours et 25 jours sont Monte Carlo ?

Reponse attendue :

- non ;
- extension operationnelle du modele 1 jour ;
- moyenne agregee ;
- volatilite, VaR et ES par scaling en racine de l'horizon ;
- prudence sur l'interpretation.

Exemple :

> Non. Ce sont des extensions operationnelles du cadre 1 jour, utiles pour donner un ordre de grandeur du risque cumule.

## Signaux d'une mauvaise reponse

La reponse est mauvaise si elle :

- contient des placeholders ;
- transforme le HMM en tendance de marche ;
- dit que la VaR est une perte maximale garantie ;
- parle de Monte Carlo alors que ce n'est pas le cas ;
- donne un conseil d'achat ou de vente ;
- invente un poids courant absent du dashboard ;
- recopie les titres markdown ;
- annonce qu'elle utilise le contexte ou les faits internes.
