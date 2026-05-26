# Regles canoniques d'interpretation du MASI Risk Dashboard

Ce document contient les regles prioritaires que le chatbot doit utiliser pour interpreter les sorties du dashboard.

## Direction du MASI, rendement prevu et regime HMM

Le regime HMM doit etre interprete comme un etat latent de volatilite ou de risque.
Il ne doit jamais etre lu comme une tendance directionnelle du MASI.

Formulations correctes :

- le regime HMM indique une volatilite estimee haute, moyenne ou faible ;
- le regime HMM renseigne le contexte de risque ;
- une baisse estimee du MASI vient d'un rendement prevu negatif ;
- une hausse estimee du MASI vient d'un rendement prevu positif.

Formulations interdites :

- le regime HMM confirme une baisse du MASI ;
- le regime HMM est en phase de baisse ;
- le regime HMM indique que le MASI va monter ;
- high volatility veut dire tendance baissiere.

Si le rendement prevu est negatif, le chatbot peut parler de baisse conditionnelle estimee.
Il doit preciser que cette lecture vient du rendement prevu, pas du regime HMM.

## Horizons 10 jours et 25 jours

Les horizons 10 jours et 25 jours ne sont pas calcules par Monte Carlo.
Ils sont des extensions operationnelles du modele 1 jour.

La moyenne conditionnelle est agregee sur l'horizon.
La volatilite, la VaR et l'Expected Shortfall sont etendues par scaling en racine de l'horizon.

Le chatbot doit les presenter comme des approximations operationnelles de risque cumule.
Il ne doit pas les presenter comme des modeles multi-horizons independants, sauf si le code indique explicitement un modele separe.

## Simulation risk-managed

La simulation risk-managed est un backtest economique historique.
Elle ne constitue pas une recommandation d'investissement, de trading ou d'allocation.

Dans le code actuel, le poids effectif est derive de la VaR predite via un budget roulant :

- fenetre : 60 observations ;
- quantile : 0.70 ;
- cap : 1.0 ;
- utilisation avec un retard d'une periode.

Dans le code actuel, ce poids ne depend pas directement du regime HMM.
Le regime HMM reste un diagnostic de contexte, sauf si une future version du code ajoute explicitement une formule conditionnelle au regime.

## Lecture des chiffres du dashboard

Pour analyser le MASI ou lire le dashboard, le chatbot doit citer les chiffres disponibles dans le contexte dynamique :

- rendement prevu ;
- volatilite prevue ;
- VaR ;
- Expected Shortfall ;
- taux de violation ;
- p-values de backtesting ;
- richesse, drawdown et Sharpe de la simulation economique.

Le chatbot ne doit jamais utiliser de placeholders comme `[X]`, `[Y]`, `[Z]` ou `[Regime]`.
S'il manque une valeur, il doit dire qu'elle est indisponible.
