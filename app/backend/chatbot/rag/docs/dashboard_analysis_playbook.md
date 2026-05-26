# Playbook d'analyse du dashboard MASI

Ce document de RAG sert a guider le chatbot lorsqu'un utilisateur demande :

- "analyse le MASI" ;
- "lis moi le dashboard" ;
- "analyse bloc par bloc" ;
- "que disent les chiffres actuels ?" ;
- "est-ce qu'on estime une baisse ou une hausse ?".

Le chatbot doit utiliser les chiffres du contexte dynamique et non produire une fiche theorique.

## Ordre de lecture recommande

### Bloc 1 : prevision courante

Lire en priorite :

- date du run ;
- date cible ;
- rendement prevu ;
- volatilite prevue ;
- VaR 5% ;
- Expected Shortfall 5% ;
- regime HMM estime.

Interpretation :

- le rendement prevu donne le sens conditionnel estime ;
- un rendement prevu negatif suggere une baisse conditionnelle ;
- un rendement prevu positif suggere une hausse conditionnelle ;
- la volatilite, la VaR et l'ES donnent l'intensite du risque ;
- le regime HMM renseigne le contexte de volatilite, pas la direction.

Formulation correcte :

> Le rendement prevu a 1 jour est negatif, ce qui suggere une baisse conditionnelle du MASI. Le regime HMM indique un contexte de volatilite, mais il ne confirme pas la direction du marche.

Formulation interdite :

> Le regime HMM confirme une baisse du MASI.

### Bloc 2 : horizons 10 jours et 25 jours

Lire :

- rendement prevu 10 jours ;
- VaR 10 jours ;
- ES 10 jours ;
- rendement prevu 25 jours ;
- VaR 25 jours ;
- ES 25 jours.

Interpretation :

- ces horizons donnent un ordre de grandeur du risque cumule ;
- ils ne doivent pas etre surinterpretes comme des previsions independantes ;
- ils ne sont pas issus d'une simulation Monte Carlo ;
- ils sont des extensions operationnelles du modele 1 jour.

Formulation correcte :

> Les horizons 10 jours et 25 jours prolongent le signal 1 jour pour donner un ordre de grandeur du risque cumule.

Formulation interdite :

> Les horizons 10 jours et 25 jours sont generes par Monte Carlo.

### Bloc 3 : backtesting statistique

Lire :

- nombre d'observations ;
- nombre de violations ;
- taux de violation ;
- taux attendu ;
- p-value de Kupiec ;
- p-value d'independance ou de couverture conditionnelle de Christoffersen ;
- diagnostic ES si disponible.

Interpretation :

- un taux de violation proche de 5% indique une calibration VaR raisonnable au niveau 5% ;
- une p-value elevee ne prouve pas que le modele est parfait ;
- une p-value elevee signifie seulement qu'on ne rejette pas clairement le test ;
- le diagnostic ES complete la VaR en regardant la severite moyenne des pertes en queue.

Formulation correcte :

> Le taux de violation est proche du niveau attendu de 5%, et les p-values ne signalent pas de rejet clair de la calibration VaR sur cette periode.

Formulation interdite :

> Le modele est parfaitement valide.

### Bloc 4 : backtesting economique

Lire :

- richesse finale risk-managed ;
- richesse finale Buy & Hold ;
- max drawdown risk-managed ;
- max drawdown Buy & Hold ;
- Sharpe risk-managed ;
- Sharpe Buy & Hold ;
- poids moyen et turnover si disponibles.

Interpretation :

- la richesse compare la trajectoire cumulee de la simulation et du benchmark ;
- le drawdown mesure la profondeur historique des pertes ;
- le Sharpe mesure le rendement par unite de volatilite ;
- une meilleure protection du drawdown ne garantit pas une richesse finale superieure ;
- cette simulation n'est pas une recommandation d'allocation.

Formulation correcte :

> La simulation risk-managed reduit le drawdown et ameliore le Sharpe, mais elle peut rester inferieure a Buy & Hold en richesse finale.

Formulation interdite :

> Il faut utiliser cette allocation maintenant.

### Bloc 5 : dernieres lignes de test

Lire :

- date ;
- rendement realise ;
- rendement predit ;
- VaR ;
- ES ;
- violation ou non.

Interpretation :

- comparer le rendement realise avec le rendement predit ;
- verifier si le rendement realise passe sous la VaR ;
- rappeler que les rendements sont bruites en finance ;
- ne pas juger tout le modele sur trois observations.

## Synthese attendue

Une bonne synthese doit tenir en 3 a 5 phrases :

- direction conditionnelle issue du rendement prevu ;
- intensite du risque issue de la VaR/ES ;
- calibration statistique issue du backtesting ;
- resultat economique issu de wealth/drawdown/Sharpe ;
- limite principale.

Exemple :

> Le MASI presente un signal de baisse conditionnelle car le rendement prevu est negatif. Le risque de queue reste mesure par la VaR et l'ES, avec un regime HMM qui renseigne la volatilite plutot que la direction. Le backtest statistique est proche de la cible de violation, ce qui soutient une calibration VaR raisonnable sur la periode. Economiquement, la simulation risk-managed ameliore certains indicateurs de risque mais ne doit pas etre lue comme une recommandation d'allocation.
