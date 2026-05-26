# Questions utilisateur frequentes et intentions

Ce document aide le RAG a relier les formulations naturelles de l'utilisateur aux bonnes regles d'interpretation.

## "On estime une baisse dans le MASI ?"

Intentions possibles :

- comprendre la direction du forecast ;
- savoir si le rendement prevu est negatif ;
- distinguer rendement prevu et regime HMM.

Reponse :

- utiliser le rendement prevu ;
- si le rendement prevu est negatif, parler de baisse conditionnelle ;
- ne jamais dire que le HMM confirme la baisse ;
- rappeler que c'est une prevision conditionnelle, pas une certitude.

## "Le regime veut dire quoi ?"

Intentions possibles :

- comprendre high_volatility, low_volatility, medium_volatility ;
- savoir si le regime est une configuration ;
- savoir si le regime donne une direction.

Reponse :

- le regime est une classification estimee par HMM ;
- il porte sur la volatilite ou le risque ;
- il ne dit pas si le MASI va monter ou baisser ;
- il peut changer avec les donnees.

## "Le poids depend du regime ?"

Intentions possibles :

- comprendre la simulation risk-managed ;
- savoir si le HMM pilote l'allocation ;
- verifier la formule actuelle du code.

Reponse :

- dans le code actuel, le poids ne depend pas directement du regime HMM ;
- le poids depend de la VaR predite via un budget roulant ;
- le poids est retarde d'une periode ;
- ce poids est une simulation, pas une allocation reelle.

## "Pourquoi la richesse risk-managed est plus faible mais Sharpe meilleur ?"

Intentions possibles :

- comprendre wealth, Sharpe et drawdown ;
- comparer risk-managed et Buy & Hold.

Reponse :

- le Sharpe mesure le rendement par unite de volatilite ;
- une richesse finale plus faible peut coexister avec un Sharpe plus eleve si la volatilite ou le drawdown diminuent ;
- il faut lire wealth, drawdown et Sharpe ensemble ;
- aucun indicateur seul ne valide une strategie reelle.

## "Les horizons 10j et 25j sont fiables ?"

Intentions possibles :

- comprendre la robustesse des horizons etendus ;
- savoir s'ils sont entraines separement.

Reponse :

- ils sont des extensions operationnelles du modele 1 jour ;
- ils donnent un ordre de grandeur ;
- l'incertitude augmente avec l'horizon ;
- ils ne doivent pas etre interpretes comme des modeles independants sauf preuve explicite.

## "Le backtest est-il a la hauteur ?"

Intentions possibles :

- evaluer la calibration VaR ;
- evaluer la calibration ES ;
- evaluer la performance economique.

Reponse :

- lire le taux de violation ;
- comparer au taux attendu ;
- lire Kupiec et Christoffersen ;
- lire le diagnostic ES ;
- lire wealth, drawdown, Sharpe ;
- conclure avec prudence.

## "Compare forecast et realised"

Intentions possibles :

- comparer rendement predit et rendement realise ;
- verifier les breaches VaR ;
- commenter les dernieres dates.

Reponse :

- citer les dernieres lignes ;
- comparer predit vs realise ;
- indiquer breach oui/non ;
- rappeler que les rendements sont bruites.

## "La page de forecasting"

Intentions possibles :

- lire le bloc de prevision actuel du dashboard ;
- comprendre les horizons 1 jour, 10 jours et 25 jours ;
- obtenir les dates cibles et les mesures rendement/risque affichees.

Reponse :

- utiliser les lignes de forecast du contexte dynamique ;
- citer les dates cibles exactes ;
- citer rendement prevu, volatilite, VaR 5% et ES 5% si disponibles ;
- ne pas inventer des valeurs pour aujourd'hui, demain ou la semaine prochaine ;
- rappeler que les horizons 10 et 25 jours sont des extensions operationnelles du cadre 1 jour ;
- rappeler que le regime HMM est un contexte de volatilite, pas une direction du MASI.

## "Est-ce que je dois acheter ou vendre ?"

Intentions possibles :

- demander un conseil d'investissement ;
- transformer le dashboard en signal de trading.

Reponse :

- refuser le conseil d'investissement ;
- recentrer sur l'interpretation du risque ;
- ne pas donner d'allocation reelle.
