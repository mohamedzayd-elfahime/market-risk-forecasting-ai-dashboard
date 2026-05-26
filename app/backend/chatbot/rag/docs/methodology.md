# Methodologie du MASI Risk Dashboard

## Objet scientifique du systeme

Le MASI Risk Dashboard est un systeme d'analyse du rendement conditionnel et du risque de baisse conditionnel applique au marche actions marocain via la serie du MASI. Son objectif est de produire une lecture rigoureuse des previsions de rendement, de volatilite et de risque de queue, et non de fournir une recommandation d'investissement.

## Serie analysee

La variable fondamentale analysee est le rendement logarithmique journalier du MASI :

$$
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
$$

ou $P_t$ designe le niveau de l'indice MASI a la date $t$.

Le systeme travaille sur les rendements plutot que sur les niveaux de prix, ce qui est standard en econometrie financiere pour modeliser la dynamique conditionnelle et les risques de queue.

## Portee du modele

L'horizon principal du systeme est un jour a l'avance. Les objets centraux sont :

- un rendement conditionnel prevu ;
- une volatilite conditionnelle prevue ;
- une Value-at-Risk conditionnelle ;
- un Expected Shortfall conditionnel ;
- un regime de volatilite latent.

Les horizons 10 jours et 25 jours doivent etre interpretes comme des extensions mathematiques operationnelles du cadre 1 jour, sauf si un modele multi-horizon independamment entraine est explicitement disponible.

## Architecture generale

Le systeme repose sur trois briques principales :

- une brique EGARCH pour la moyenne et la volatilite conditionnelles ;
- une brique LSTM quantile pour la prevision directe de la VaR ;
- une brique Ridge en deux etapes pour la prevision de l'Expected Shortfall.

Une brique HMM est utilisee en parallele pour la classification de regime de volatilite.

## Bloc EGARCH : moyenne et volatilite conditionnelles

### Equation de decomposition

Le cadre de base retenu pour le rendement est :

$$
r_{t+1} = \mu_{t+1|t} + \sigma_{t+1|t} z_{t+1}
$$

ou :

- $\mu_{t+1|t}$ est la moyenne conditionnelle a un jour ;
- $\sigma_{t+1|t}$ est la volatilite conditionnelle a un jour ;
- $z_{t+1}$ est une innovation standardisee.

### Role du bloc EGARCH

Le bloc EGARCH fournit :

- une prevision de moyenne conditionnelle ;
- une prevision de volatilite conditionnelle ;
- des variables de regime et de residu standardise utiles pour l'analyse du risque.

Ces sorties sont exploitees a la fois pour l'interpretation economique et pour la branche hybride de prevision du rendement.

## Prevision du rendement : hybridation EGARCH-LSTM

### Principe

Le rendement a 1 jour n'est pas lu comme une simple moyenne lisse. Le systeme utilise une hybridation entre la structure EGARCH et une prevision LSTM de l'innovation standardisee.

La cible d'apprentissage est :

$$
z_{t+1} = \frac{r_{t+1} - \mu_{t+1|t}}{\sigma_{t+1|t}}
$$

Le LSTM apprend donc a predire :

$$
\widehat{z}_{t+1|t}
$$

puis la prevision finale du rendement est reconstruite par :

$$
\widehat{r}_{t+1|t} = \mu_{t+1|t} + \sigma_{t+1|t}\widehat{z}_{t+1|t}
$$

### Interpretation

Cette structure permet :

- de conserver une base econometrique interpretable via $\mu_{t+1|t}$ et $\sigma_{t+1|t}$ ;
- de laisser au LSTM le soin d'apprendre la composante residuelle non lineaire ;
- d'eviter une prevision de rendement excessivement lisse ou excessivement erratique.

## Prevision de la Value-at-Risk

### Principe

La VaR a un jour est predite directement par un modele LSTM quantile. La cible est le rendement futur lui-meme, et non un residu re-normalise.

La VaR est donc estimee directement comme quantile conditionnel inferieur :

$$
\widehat{\mathrm{VaR}}_{\alpha,t+1|t}
=
\widehat{Q}_{\alpha}(r_{t+1} \mid \mathcal{F}_t)
$$

### Variables utilisees pour la VaR

La branche VaR utilise uniquement les variables de base suivantes :

- rendement log journalier ;
- volatilite glissante a 5 jours ;
- volatilite glissante a 21 jours ;
- moyenne glissante a 21 jours.

### Fonction de perte

Le modele est entraine avec une pinball loss au niveau $\alpha$ :

$$
L_\alpha(y,\hat y) = \max\left(\alpha(y-\hat y),(\alpha-1)(y-\hat y)\right)
$$

Cette fonction force le modele a apprendre directement le quantile de queue cherche.

## Prevision de l'Expected Shortfall

### Approche two-step

L'Expected Shortfall est estime en deux etapes :

1. prevision de la VaR ;
2. estimation du shortfall conditionnel au-dela de la VaR.

Le shortfall est defini par :

$$
\text{shortfall}_{t+1} = \max\left(\widehat{\mathrm{VaR}}_{\alpha,t+1|t} - r_{t+1}, 0\right)
$$

Un modele Ridge est ensuite ajuste sur les seules observations de violation afin de predire ce shortfall.

La prevision finale d'ES est :

$$
\widehat{\mathrm{ES}}_{\alpha,t+1|t}
=
\widehat{\mathrm{VaR}}_{\alpha,t+1|t}
-
\widehat{\text{shortfall}}_{t+1|t}
$$

avec la contrainte economiquement coherente :

$$
\widehat{\mathrm{ES}}_{\alpha,t+1|t} \le \widehat{\mathrm{VaR}}_{\alpha,t+1|t}
$$

### Interpretation

L'ES doit etre lu comme une mesure de severite moyenne conditionnelle des pertes au-dela du seuil de VaR.

## Regime de volatilite via HMM

Un Hidden Markov Model est ajuste uniquement sur le bloc d'apprentissage afin d'identifier des regimes latents de volatilite.

Le regime courant doit etre interprete comme une classification statistique conditionnelle, par exemple :

- regime de faible volatilite ;
- regime de volatilite intermediaire ;
- regime de forte volatilite.

Il ne constitue pas un diagnostic structurel exhaustif du marche.

## Construction des horizons 10 jours et 25 jours

### Principe general

Les horizons 10 jours et 25 jours ne sont pas issus d'un apprentissage multi-horizon independant. Ce sont des extensions operationnelles du cadre 1 jour.

### Rendement

Pour les horizons 10 jours et 25 jours, le rendement affiche est aligne sur la moyenne conditionnelle agregee :

$$
\mu_h \approx h \mu_1
$$

Autrement dit, la presentation multi-jours du rendement repose sur l'agregation de la moyenne conditionnelle, et non sur une extrapolation brute du signal d'innovation 1 jour.

### VaR et ES

Les mesures de risque de queue multi-jours sont etendues de maniere operationnelle par une loi en racine de l'horizon :

$$
\mathrm{VaR}_h \approx \sqrt{h}\,\mathrm{VaR}_1
$$

$$
\mathrm{ES}_h \approx \sqrt{h}\,\mathrm{ES}_1
$$

Ces quantites doivent etre interpretees comme des prolongements de risque et non comme des previsions structurelles multi-horizons pleinement re-entrainees.

## Reconstruction indicative d'un niveau de prix

Lorsqu'un proxy de prix est affiche a partir d'un rendement prevu, il est reconstruit selon :

$$
\widehat{P}_{t+h|t} = P_t \exp(\widehat{r}_{t+h|t})
$$

Cette transformation a un role visuel et pedagogique. Elle ne doit pas etre interpretee comme une prevision structurelle complete de la trajectoire future des points d'indice.

## Backtesting statistique

### Violations de VaR

Une violation de VaR est definie par :

$$
I_{t+1} = \mathbf{1}\{r_{t+1} < \widehat{\mathrm{VaR}}_{\alpha,t+1|t}\}
$$

Le taux empirique de violation est :

$$
\widehat{\pi} = \frac{1}{T}\sum_{t=1}^{T} I_t
$$

### Tests de calibration

Le systeme mobilise notamment :

- un test de couverture inconditionnelle de Kupiec ;
- un test d'independance de Christoffersen ;
- un test de couverture conditionnelle ;
- un controle de calibration de queue pour l'ES.

### Residuel moyen de l'ES

La calibration moyenne de l'ES en queue peut etre suivie via :

$$
\frac{1}{N_{tail}} \sum_{t \in \mathcal{T}_{tail}} \left(r_t - \widehat{\mathrm{ES}}_t\right)
$$

ou $\mathcal{T}_{tail}$ represente l'ensemble des observations pour lesquelles il y a violation de VaR.

## Backtesting economique

### Regle simulee

L'evaluation economique repose sur une simulation historique d'une regle de gestion du risque fondee sur le niveau prevu de VaR. Une ponderation est construite a partir d'un budget de risque derive de la VaR prevue.

La regle peut s'ecrire schematiquement :

$$
w_t = \mathrm{clip}\left(\frac{\mathrm{budget}_t}{|\widehat{\mathrm{VaR}}_t|}, 0, 1\right)
$$

avec utilisation effective du poids avec un retard d'une periode pour eviter tout biais d'anticipation :

$$
w_t^{used} = w_{t-1}
$$

### Rendement et richesse de la simulation

Le rendement simple de l'actif est :

$$
R_t = e^{r_t} - 1
$$

Le rendement simple de la simulation risk-managed est :

$$
R_t^{strat} = w_{t-1} R_t - c \cdot turnover_t
$$

ou $c$ represente un cout de transaction eventuel.

La richesse evolue ensuite selon :

$$
W_t^{strat} = W_{t-1}^{strat}(1 + R_t^{strat})
$$

et le Buy & Hold selon :

$$
W_t^{BH} = W_{t-1}^{BH}(1 + R_t)
$$

### Drawdown

Le drawdown est defini par :

$$
DD_t = \frac{W_t}{\max_{s \le t} W_s} - 1
$$

et le maximum drawdown par :

$$
\min_t DD_t
$$

Le backtest economique ne constitue pas une recommandation d'allocation. Il sert uniquement a mesurer l'utilite economique historique des previsions de risque sous une regle simulee.

## Controle de leakage

Le pipeline est construit selon une logique causale stricte :

- split chronologique entre apprentissage, validation et test ;
- sequences LSTM construites uniquement a partir du passe ;
- variables EGARCH causales ;
- HMM ajuste uniquement sur l'echantillon d'apprentissage ;
- poids economiques utilises avec un decalage d'une periode.

Ainsi, les previsions a la date $t$ sont construites uniquement a partir de l'information disponible jusqu'a $t$.

## Nature du systeme

Le MASI Risk Dashboard est un outil d'interpretation quantitative du risque de marche marocain. Il ne constitue ni un systeme de recommandation de titres, ni un conseil de portefeuille, ni une instruction de trading.
