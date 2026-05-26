# Glossaire du MASI Risk Dashboard

## MASI -- Moroccan All Shares Index

Le **MASI** désigne le **Moroccan All Shares Index**, un indice large associé au marché actions marocain et à la Bourse de Casablanca. Dans ce projet, il est traité comme une **série de référence du risque agrégé du marché actions marocain**.

Le MASI est utilisé comme proxy du risque global du marché actions marocain. L’objectif du dashboard n’est pas d’analyser le risque idiosyncratique d’une seule société cotée, mais de suivre et prévoir le risque conditionnel associé à l’évolution globale du marché.

Le dashboard peut afficher les niveaux historiques du MASI, notés \(P_t\), afin de fournir une lecture visuelle de l’évolution du marché. Cependant, la couche de prévision ne modélise pas directement les points bruts de l’indice. Les modèles de forecasting travaillent principalement sur les **rendements logarithmiques journaliers** :

\[
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
\]

où \(P_t\) désigne le niveau de l’indice MASI à la date \(t\), et \(r_t\) le rendement logarithmique journalier.

Le système prévoit donc un **rendement conditionnel** et des **mesures de risque conditionnelles**, et non un niveau futur exact de l’indice.

## Prix affichés et rendements modélisés

Le dashboard combine deux niveaux de lecture :

1. une lecture descriptive, où les graphiques peuvent afficher les prix ou niveaux historiques du MASI ;
2. une lecture prédictive, où les modèles travaillent sur les rendements logarithmiques, la volatilité, la VaR et l’Expected Shortfall.

Les prix affichés servent principalement à contextualiser l’évolution historique du marché. Ils permettent de visualiser les phases de hausse, de baisse, de stress ou de stabilisation du MASI.

En revanche, les prévisions du modèle portent sur les rendements :

\[
\hat{r}_{t+1|t}
\]

et sur les mesures de risque associées à la distribution conditionnelle de ces rendements :

\[
\widehat{\operatorname{VaR}}_{\alpha,t+1|t},
\qquad
\widehat{\operatorname{ES}}_{\alpha,t+1|t},
\qquad
\hat{\sigma}_{t+1|t}
\]

Ainsi, lorsqu’un graphique présente le niveau du MASI, il s’agit d’une information descriptive sur le marché. Lorsqu’un graphique présente une prévision de rendement, une VaR, une ES ou une volatilité, il s’agit d’une information exprimée dans l’espace des rendements et du risque conditionnel.

Une prévision de rendement ne doit donc pas être interprétée comme une prévision directe et exacte du futur niveau du MASI.

## Indice boursier versus ETF

Un **indice boursier** est une mesure statistique qui résume l’évolution d’un marché ou d’un segment de marché. Il sert de baromètre de performance et de risque, mais n’est pas en lui-même un titre directement achetable.

Un **ETF** est un fonds coté, négocié en bourse, qui cherche généralement à reproduire un indice, un panier d’actifs ou une stratégie.

Dans ce projet, le **MASI est utilisé comme série d’indice**, et non comme un ETF. Le dashboard ne fournit pas d’instructions de négociation sur des ETF ni sur des titres individuels.

## Marché actions marocain

Le marché actions marocain est représenté dans ce projet par le MASI. Le risque étudié correspond donc au risque agrégé du marché coté marocain, et non au risque spécifique d’une entreprise.

Ce marché peut présenter des caractéristiques importantes pour la modélisation du risque : liquidité variable, concentration sectorielle, périodes de faible activité, chocs macroéconomiques, effets de calendrier et changements de régime.

Ces caractéristiques justifient l’utilisation de modèles capables de représenter une volatilité variable dans le temps et des épisodes de stress.

## Rendement logarithmique

Le rendement logarithmique journalier est défini par :

\[
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
= \ln(P_t) - \ln(P_{t-1})
\]

Les modèles de risque financiers travaillent généralement sur les rendements plutôt que sur les niveaux d’indice, car les rendements sont plus adaptés à l’analyse statistique de la dynamique conditionnelle, de la volatilité et des queues de distribution.

Dans ce dashboard, les pertes sont interprétées à travers les rendements négatifs. Un rendement négatif indique une baisse relative du niveau de l’indice entre deux dates.

## Prévision de rendement

La prévision de rendement à un jour est notée :

\[
\hat{r}_{t+1|t}
\]

Elle représente une estimation du rendement attendu pour la prochaine séance, conditionnellement à l’information disponible jusqu’à la date \(t\).

Il s’agit d’une prévision conditionnelle, pas d’une certitude. Une prévision positive n’exclut pas une perte réalisée, et une prévision négative n’implique pas nécessairement une baisse certaine.

Dans les séries financières, la moyenne conditionnelle des rendements est souvent plus difficile à prévoir que la volatilité ou le risque de queue.

## Information disponible

L’information disponible à la date \(t\) est notée :

\[
\mathcal{F}_t
\]

Elle représente l’ensemble des informations utilisables au moment de produire la prévision : rendements passés, variables explicatives disponibles, volatilité estimée, signaux de régime et sorties de modèles calculées sans utiliser le futur.

Une prévision rigoureuse doit respecter le principe de non-fuite d’information : aucune donnée future ne doit être utilisée pour prévoir \(t+1\).

## Rendement attendu conditionnel

Le **rendement attendu conditionnel** représente la moyenne prévue du rendement à court terme compte tenu de l’information disponible à la date \(t\) :

\[
\mu_{t+1|t} = \mathbb{E}[r_{t+1} \mid \mathcal{F}_t]
\]

Il s’agit d’une **espérance conditionnelle**, pas d’une certitude. Une moyenne positive n’exclut pas une perte réalisée, et une moyenne négative n’implique pas qu’une baisse se produira avec certitude.

## Volatilité conditionnelle

La **volatilité conditionnelle** mesure l’intensité du risque de fluctuation attendue, conditionnellement à l’information disponible :

\[
\sigma_{t+1|t} =
\sqrt{\operatorname{Var}(r_{t+1} \mid \mathcal{F}_t)}
\]

Une hausse de volatilité conditionnelle signale une augmentation de l’incertitude sur les rendements futurs. Elle indique un environnement de risque plus tendu, mais ne donne pas directement la direction du marché.

Une volatilité élevée signifie une dispersion plus forte des rendements possibles, pas nécessairement une baisse attendue.

## Volatility clustering

Le **volatility clustering** désigne le fait que les grandes variations de marché tendent à être suivies par d’autres grandes variations, tandis que les périodes calmes tendent à être suivies par d’autres périodes calmes.

Cette propriété empirique des séries financières justifie l’utilisation de modèles de volatilité conditionnelle, notamment les modèles de type GARCH.

## GARCH-type models

Un cadre général de modélisation des rendements financiers peut s’écrire :

\[
r_t = \mu_t + \varepsilon_t,
\qquad
\varepsilon_t = \sigma_t z_t
\]

où :

- \(\mu_t\) est la moyenne conditionnelle ;
- \(\sigma_t\) est la volatilité conditionnelle ;
- \(z_t\) est une innovation standardisée ;
- \(\varepsilon_t\) est le choc non expliqué par la moyenne conditionnelle.

Les modèles de type GARCH cherchent à représenter la dynamique de \(\sigma_t\), c’est-à-dire la manière dont le risque varie dans le temps.

## EGARCH

Le modèle **EGARCH** est une spécification de volatilité conditionnelle dans laquelle on modélise le logarithme de la variance conditionnelle. Une écriture générale est :

\[
\log(\sigma_t^2)
=
\omega
+
\sum_{i=1}^{p}\beta_i \log(\sigma_{t-i}^2)
+
\sum_{j=1}^{q}\alpha_j
\left(|z_{t-j}|-\mathbb{E}|z_{t-j}|\right)
+
\sum_{k=1}^{o}\gamma_k z_{t-k}
\]

avec :

\[
z_t = \frac{\varepsilon_t}{\sigma_t}
\]

L’intérêt économétrique d’EGARCH est double. Premièrement, la modélisation du logarithme de la variance garantit une variance positive. Deuxièmement, le modèle peut capturer des effets d’asymétrie, c’est-à-dire des réactions différentes de la volatilité selon le signe des chocs.

Dans le dashboard, les sorties EGARCH servent à enrichir l’interprétation du risque conditionnel, notamment à travers la moyenne conditionnelle, la volatilité conditionnelle et la dynamique récente du marché.

## AR(1)

Un modèle **AR(1)**, ou autorégressif d’ordre 1, suppose que la moyenne conditionnelle dépend du rendement précédent :

\[
r_t = c + \phi r_{t-1} + \varepsilon_t
\]

Le coefficient \(\phi\) mesure la persistance linéaire de court terme. Dans la couche d’interprétation, cela signifie qu’une composante de la moyenne peut refléter une inertie statistique des rendements récents, sans constituer une promesse de continuation mécanique.

## Résidu standardisé

Le **résidu standardisé** est le choc centré et mis à l’échelle par la volatilité conditionnelle :

\[
z_t = \frac{\varepsilon_t}{\sigma_t}
\]

Il sert à diagnostiquer si les mouvements observés sont grands ou petits relativement au niveau de risque conditionnel estimé.

## Hidden Markov Model / HMM

Un **Hidden Markov Model**, ou HMM, est un modèle à états latents. Il suppose que le marché peut être décrit par un état non directement observable :

\[
S_t \in \{1,\ldots,K\}
\]

La dynamique de passage entre les régimes est décrite par des probabilités de transition :

\[
\Pr(S_t=j \mid S_{t-1}=i)=p_{ij}
\]

où \(p_{ij}\) représente la probabilité de passer du régime \(i\) au régime \(j\).

Dans le dashboard, le HMM sert à classifier l’environnement de volatilité. Les labels comme « faible volatilité », « volatilité moyenne » ou « haute volatilité » sont des interprétations statistiques des états latents.

Ils ne constituent pas des prévisions déterministes ni des signaux de trading.

## Régime de volatilité

Un **régime de volatilité** est une classification statistique de l’environnement de risque courant.

Un régime de faible volatilité correspond généralement à un environnement plus calme. Un régime de volatilité moyenne correspond à une incertitude intermédiaire. Un régime de haute volatilité correspond à une incertitude plus forte et à une sensibilité accrue au risque de baisse.

Le régime affiché par le dashboard doit être interprété comme une information de contexte sur le risque, pas comme une recommandation financière.

## Value-at-Risk / VaR

La **Value-at-Risk** au niveau \(\alpha\) est un quantile de la distribution conditionnelle des rendements. Pour un horizon d’un jour :

\[
\operatorname{VaR}_{\alpha,t+1|t}
=
Q_{\alpha}(r_{t+1}\mid\mathcal{F}_t)
\]

ou, de manière équivalente :

\[
\Pr(r_{t+1}\leq
\operatorname{VaR}_{\alpha,t+1|t}
\mid \mathcal{F}_t)=\alpha
\]

Pour \(\alpha=5\%\), la VaR correspond au quantile conditionnel à 5 % du rendement du lendemain.

Dans un cadre de rendements, une VaR plus négative indique un seuil de perte potentielle plus sévère dans la queue gauche de la distribution. La VaR ne décrit pas la taille moyenne des pertes au-delà de ce seuil.

La VaR est probabiliste. Elle n’est pas une perte maximale garantie.

## Expected Shortfall / ES

L’**Expected Shortfall**, ou **ES**, mesure la perte moyenne conditionnelle une fois que la zone de queue a été franchie. Pour un seuil \(\alpha\) :

\[
\operatorname{ES}_{\alpha,t+1|t}
=
\mathbb{E}
\left[
r_{t+1}
\mid
r_{t+1}\leq
\operatorname{VaR}_{\alpha,t+1|t},
\mathcal{F}_t
\right]
\]

L’ES mesure donc la sévérité moyenne des pertes conditionnellement au dépassement de la VaR.

Dans une distribution de rendements, l’ES est généralement plus négatif que la VaR dans la queue gauche. La VaR indique où commence la zone de pertes extrêmes, tandis que l’ES indique la perte moyenne attendue dans cette zone.

## Violation de VaR

Une **violation de VaR** se produit lorsque le rendement réalisé est plus défavorable que le seuil de VaR :

\[
I_{t+1}
=
\mathbf{1}
\left\{
r_{t+1}
<
\operatorname{VaR}_{\alpha,t+1|t}
\right\}
\]

Si \(I_{t+1}=1\), cela signifie que la perte réalisée dépasse le seuil prévu par la VaR.

Pour une VaR à 5 %, un modèle bien calibré en couverture inconditionnelle devrait produire environ 5 % de violations sur un grand échantillon de backtest.

## Taux de violation

Le **taux de violation** est la proportion empirique de violations observées sur un échantillon de backtest :

\[
\widehat{\pi}
=
\frac{1}{T}
\sum_{t=1}^{T} I_t
\]

Ce taux est comparé au niveau cible \(\alpha\). Une valeur proche de \(\alpha\) est nécessaire pour juger la VaR correctement calibrée en moyenne.

Cependant, cela ne suffit pas. Les violations doivent aussi être indépendantes dans le temps. Un modèle peut avoir le bon nombre total de violations, mais échouer si ces violations sont regroupées en périodes de stress.

## Test de Kupiec

Le **test de Kupiec** évalue la couverture inconditionnelle de la VaR. Il teste si la fréquence observée des violations est cohérente avec le niveau théorique \(\alpha\).

L’hypothèse nulle est :

\[
H_0:\Pr(I_t=1)=\alpha
\]

Si le test de Kupiec n’est pas rejeté, cela suggère que le nombre total de violations est statistiquement compatible avec le niveau attendu.

Cependant, réussir le test de Kupiec ne prouve pas que le modèle est pleinement valide, car ce test ne vérifie pas l’indépendance temporelle des violations.

## Test d’indépendance de Christoffersen

Le **test d’indépendance de Christoffersen** évalue si les violations de VaR sont indépendantes dans le temps.

L’hypothèse nulle est :

\[
H_0: I_t \text{ est indépendant dans le temps}
\]

Si les violations sont regroupées, cela peut indiquer que le modèle ne capture pas correctement les dynamiques de stress.

Un modèle peut passer le test de Kupiec tout en échouant au test d’indépendance de Christoffersen.

## Couverture conditionnelle

La **couverture conditionnelle** combine deux exigences :

1. le bon nombre moyen de violations ;
2. l’indépendance temporelle des violations.

Un modèle peut avoir une couverture inconditionnelle acceptable, mais une mauvaise couverture conditionnelle si les violations apparaissent en clusters.

La couverture conditionnelle est donc un critère plus strict que la simple fréquence moyenne des violations.

## Backtesting VaR

Un **backtest statistique de VaR** examine si la fréquence et la structure temporelle des violations sont cohérentes avec la VaR produite par le modèle.

Son rôle est diagnostique :

- vérifier la couverture inconditionnelle ;
- vérifier si les violations semblent indépendantes dans le temps ;
- détecter une sous-estimation ou une surestimation récurrente du risque de queue.

Le backtesting doit être réalisé hors échantillon et dans l’ordre chronologique. Une prévision produite à la date \(t\) doit être comparée au rendement effectivement observé après cette date.

## Backtesting ES

Le **backtesting de l’Expected Shortfall** est plus délicat que celui de la VaR, car l’ES concerne la sévérité moyenne des pertes conditionnellement aux événements de queue.

Une idée de diagnostic consiste à examiner les résidus de queue :

\[
e_t =
r_t - \operatorname{ES}_{\alpha,t}
\quad
\text{pour les observations telles que }
r_t \leq \operatorname{VaR}_{\alpha,t}
\]

L’objectif est de vérifier si les pertes observées au-delà de la VaR sont cohérentes avec la sévérité prédite par l’ES.

Cette analyse doit être interprétée avec prudence, car le nombre d’observations en queue peut être faible.

## Backtest économique

Un **backtest économique** ne vérifie pas seulement la calibration statistique. Il évalue aussi les conséquences économiques d’une règle de décision ou d’une allocation conditionnelle dérivée des signaux de risque.

Cette lecture doit toujours rester prudente : de bons indicateurs économiques historiques ne constituent pas une garantie de performance future.

Dans le dashboard, l’évaluation économique sert à analyser si les signaux de risque ont une utilité pratique dans une simulation historique, pas à produire des recommandations d’investissement.

## Ratio de Sharpe

Le **ratio de Sharpe** rapporte un rendement excédentaire à une mesure de volatilité :

\[
\operatorname{Sharpe}
=
\frac{\mathbb{E}[R_p - R_f]}{\sigma_p}
\]

où :

- \(R_p\) est le rendement de la stratégie ou du portefeuille ;
- \(R_f\) est le taux sans risque, s’il est utilisé ;
- \(\sigma_p\) est la volatilité du rendement.

Dans un tableau de bord de risque, le Sharpe sert à comparer des profils rendement-risque historiques ou simulés. Il ne remplace pas l’analyse des queues de distribution, des violations de VaR et des drawdowns.

## Ratio de Sortino

Le **ratio de Sortino** est une mesure de performance ajustée du risque qui pénalise principalement la volatilité défavorable :

\[
\operatorname{Sortino}
=
\frac{\mathbb{E}[R_p - R_f]}{\sigma_{\text{downside}}}
\]

où \(\sigma_{\text{downside}}\) représente la volatilité des rendements négatifs ou inférieurs à un seuil cible.

Il est utile lorsque l’on souhaite distinguer la volatilité totale de la volatilité réellement défavorable. Il ne doit être interprété que s’il est disponible dans le contexte dynamique du dashboard.

## Maximum drawdown

Le **maximum drawdown** mesure la perte maximale entre un sommet historique et le creux suivant :

\[
\operatorname{MDD}
=
\max_t
\left(
\frac{\operatorname{Peak}_t - W_t}
{\operatorname{Peak}_t}
\right)
\]

où :

\[
\operatorname{Peak}_t = \max_{s\leq t} W_s
\]

et \(W_t\) représente la richesse cumulée à la date \(t\).

Le maximum drawdown renseigne sur la profondeur des pertes cumulées. Il complète utilement les mesures de volatilité moyenne et de performance ajustée du risque.

## Wealth Curve

La **wealth curve**, ou courbe de richesse cumulée, décrit l’évolution d’un capital simulé au cours du temps :

\[
W_t = W_{t-1}(1+R_t)
\]

où \(R_t\) est le rendement de la stratégie ou du benchmark à la date \(t\).

Elle permet de comparer visuellement l’évolution cumulative d’une stratégie de gestion du risque et d’un benchmark. Elle reste une simulation historique et ne garantit pas la performance future.

## Règle de risk-targeting conditionnelle au régime

Note d'implémentation actuelle : dans le code livré du dashboard, la simulation économique affichée utilise un budget roulant fondé sur la VaR prédite et un poids retardé d'une période. Le régime HMM n'entre pas directement dans la formule effective du poids, sauf évolution explicite du code. Le chatbot doit donc décrire la version actuelle comme une règle VaR-budget, et non comme une règle conditionnelle au régime.

Une **règle de risk-targeting conditionnelle au régime** ajuste l’exposition simulée au MASI en fonction de deux éléments :

- l’intensité du risque prévu, mesurée par la VaR ;
- le régime de volatilité estimé par le HMM.

La règle générale du poids simulé peut s’écrire :

\[
w_t =
\min\left(
1,
\frac{B(S_t)}
{|\widehat{\operatorname{VaR}}_{\alpha,t+1|t}|}
\right)
\]

où :

- \(w_t\) est le poids d’exposition simulé à la date \(t\) ;
- \(S_t\) est le régime HMM estimé à la date \(t\) ;
- \(|\widehat{\operatorname{VaR}}_{\alpha,t+1|t}|\) mesure l’intensité du risque prévu pour la période suivante ;
- \(B(S_t)\) est le budget de risque conditionnel au régime.

Dans une spécification à deux régimes, le budget de risque peut être défini par des quantiles historiques de l’intensité de la VaR :

\[
B(S_t)=
\begin{cases}
Q_{0.25}\left(|\widehat{\operatorname{VaR}}_{\alpha}|\right), & \text{si } S_t \text{ correspond à un régime stressé},\\
Q_{0.75}\left(|\widehat{\operatorname{VaR}}_{\alpha}|\right), & \text{si } S_t \text{ correspond à un régime calme}.
\end{cases}
\]

Le quantile \(Q_{0.25}\) impose un budget de risque plus prudent en régime stressé, tandis que le quantile \(Q_{0.75}\) autorise un budget de risque plus permissif en régime calme.

Lorsque la VaR prévue devient plus sévère que le budget de risque du régime courant, le poids simulé diminue mécaniquement. Lorsque le risque prévu reste inférieur ou égal au budget de risque, le poids reste plafonné à \(1\).

Cette règle sert à évaluer l’utilité économique historique des prévisions de risque et des régimes HMM. Elle ne constitue pas une recommandation d’allocation réelle.

## Budget de risque conditionnel au régime

Le budget de risque conditionnel au régime, noté \(B(S_t)\), désigne le seuil de risque utilisé pour piloter l’exposition simulée dans le backtest économique.

Il n’est pas choisi comme une constante arbitraire unique. Il est calibré à partir de quantiles historiques de l’intensité de la VaR prévue, avec un niveau plus prudent en régime stressé et un niveau plus permissif en régime calme.

Cette calibration réduit l’arbitraire d’un budget fixe, mais elle reste dépendante du protocole expérimental : choix des régimes, choix des quantiles, période de calibration et mesure de risque retenue.

## Analyse financière descriptive

L’**analyse financière descriptive** désigne l’interprétation contrôlée des indicateurs de risque affichés par le dashboard.

Elle peut porter sur :

- le niveau de risque courant ;
- la sévérité de la VaR ;
- la sévérité de l’ES ;
- le régime de volatilité ;
- la calibration statistique du modèle ;
- les résultats de backtesting ;
- la comparaison historique des courbes de richesse.

Cette analyse reste descriptive et conditionnelle aux sorties du modèle. Elle ne doit pas être transformée en conseil d’investissement.

## Outil d’interprétation du risque, pas système de recommandation

Le MASI Risk Dashboard est un **outil d’interprétation du risque de marché**. Il n’est ni un système de recommandation boursière, ni un conseiller en portefeuille, ni un générateur d’ordres de trading.

Les définitions du présent glossaire servent à expliquer les sorties quantitatives du système, pas à fournir un avis d’investissement.
