# Limitations du MASI Risk Dashboard

## Nature non prescriptive du système

Le dashboard est un **outil d'analyse et d'interprétation du risque**. Il ne doit pas être présenté comme un système de recommandation d'investissement, de sélection de titres, de market timing ou de construction optimale de portefeuille.

Le chatbot doit refuser toute reformulation qui transformerait les sorties du système en conseil du type achat, vente, conservation, réduction ou augmentation d'exposition réelle.

## MASI comme série de référence, non comme actif directement tradable

Le MASI est un **indice de marché**, pas un instrument directement négociable au même titre qu'une action ou qu'un ETF. Les prévisions et mesures de risque portent sur la série de référence de l'indice, utilisée comme proxy du risque agrégé du marché actions marocain.

Toute transposition immédiate à un produit d'investissement concret exigerait des hypothèses supplémentaires qui sortent du cadre du dashboard.

## Limite entre affichage des prix et modélisation des rendements

Le dashboard peut afficher les niveaux historiques du MASI, notés \(P_t\), ainsi que plusieurs graphiques descriptifs pour faciliter la lecture visuelle du marché.

Cependant, la couche de prévision ne modélise pas directement le niveau futur exact de l'indice. Le forecasting principal est réalisé sur les rendements logarithmiques journaliers :

$$
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
$$

Les prévisions de rendement, de volatilité, de VaR, d'Expected Shortfall et les diagnostics de backtesting doivent donc être interprétés dans l'espace des rendements, et non comme des prévisions directes de points d'indice.

Si un graphique affiche les prix du MASI, il doit être lu comme un support descriptif de contexte. Si un graphique affiche une prévision, une zone de risque, une VaR ou une ES, il doit être lu comme une information relative au rendement et au risque conditionnel.

Toute conversion d'un rendement prévu vers un niveau de prix implicite dépend du dernier niveau observé \(P_t\) et reste une approximation pédagogique :

$$
\widehat{P}_{t+1|t}
=
P_t \exp(\hat{r}_{t+1|t})
$$

Cette reconstruction ne doit pas être interprétée comme un objectif de marché, une cible de prix ou une garantie de trajectoire future.

## Horizon principal à un jour

Le cœur méthodologique du système est l'horizon **un jour à l'avance**. Les horizons **10 jours** et **25 jours** ne doivent pas être surinterprétés.

Sauf présence explicite d'un modèle multi-horizon estimé indépendamment, ces horizons correspondent à des **extensions mathématiques du cadre un jour**. Ils sont utiles pour donner un ordre de grandeur du risque cumulé, mais ils ne possèdent pas nécessairement la même robustesse qu'un modèle entraîné directement à ces horizons.

Les horizons étendus doivent donc être lus comme des approximations opérationnelles, particulièrement sensibles à la volatilité conditionnelle, aux queues épaisses, aux ruptures de régime et à la dépendance temporelle.

## Risque de mauvaise lecture d'une prévision conditionnelle

Une prévision conditionnelle n'est pas une certitude. Une moyenne prévue positive peut être suivie d'une réalisation négative, et inversement. De même, une hausse de volatilité conditionnelle ne signifie pas obligatoirement une baisse de marché, mais une augmentation de l'incertitude et de l'amplitude attendue des mouvements.

Le chatbot doit donc distinguer clairement :

- la direction moyenne attendue ;
- la dispersion attendue ;
- le risque de queue ;
- le régime estimé ;
- la réalisation effectivement observée.

## Limite du trade-off rendement-risque

Le dashboard peut aider à lire le **couple rendement-risque**, mais il ne doit pas transformer ce compromis en recommandation d'action.

Une prévision de rendement favorable peut être accompagnée d'une volatilité élevée, d'une VaR sévère ou d'un Expected Shortfall important. Inversement, une prévision de rendement modérée peut être associée à un risque conditionnel plus faible.

Le trade-off doit donc être interprété comme une relation descriptive entre rendement attendu et risque estimé. Il ne suffit pas à déterminer une décision d'investissement réelle, car une telle décision dépendrait aussi d'objectifs individuels, de contraintes de portefeuille, de coûts de transaction, de liquidité, de fiscalité et de tolérance au risque.

## Limites de la VaR

La VaR est une mesure de **seuil quantile**. Elle indique à partir de quel niveau de perte la queue basse commence, mais elle ne renseigne pas à elle seule sur la sévérité moyenne des pertes au-delà de ce seuil.

Autres limites importantes :

- la VaR dépend du bon calibrage de la distribution conditionnelle ;
- elle peut être instable en période de rupture de régime ;
- elle peut donner une impression de précision excessive si elle est lue sans intervalle de prudence ;
- elle ne doit jamais être interprétée comme une perte maximale garantie.

## Limites de l'Expected Shortfall

L'Expected Shortfall apporte une information plus riche sur les pertes extrêmes, mais il demeure sensible :

- au bon modelage des queues de distribution ;
- à la rareté des observations extrêmes ;
- à l'instabilité statistique en petit échantillon ;
- au choix du modèle utilisé pour approximer la distribution conditionnelle.

Une valeur d'ES ne doit jamais être présentée comme une borne absolue des pertes.

## Limites des modèles EGARCH

Les modèles EGARCH sont puissants pour représenter la volatilité conditionnelle, mais ils reposent sur des hypothèses paramétriques et sur une structure spécifique de dépendance temporelle. Ils peuvent être mis en défaut lorsque :

- les ruptures de marché sont brutales ;
- les distributions empiriques s'écartent fortement du cadre paramétrique retenu ;
- la relation entre chocs et volatilité évolue dans le temps ;
- les queues de distribution deviennent plus sévères que prévu par le modèle.

La moyenne AR(1) et la volatilité EGARCH doivent donc être lues comme des **résumés modélisés**, pas comme des lois immuables du marché.

## Limites du régime HMM

Un HMM classe les observations dans des états latents de manière probabiliste. Cette classification est utile, mais elle simplifie fortement la réalité.

Ses limites incluent :

- une dépendance au nombre de régimes imposé ;
- un risque d'instabilité des étiquettes selon l'échantillon ;
- une interprétation économique parfois ambiguë des états latents ;
- une sensibilité aux périodes de marché utilisées pour l'estimation.

Le régime affiché doit être décrit comme un **signal de contexte statistique** plutôt que comme une vérité de structure, une cause économique directe ou un signal d'allocation réelle.

## Limites du backtesting statistique

Un backtest statistique renseigne sur la calibration passée, pas sur la garantie future. Même un résultat de backtest satisfaisant ne prouve pas qu'un modèle restera bien calibré sous des conditions de marché nouvelles.

Les conclusions de backtest peuvent aussi être fragilisées par :

- la taille d'échantillon ;
- la rareté des événements extrêmes ;
- la présence de dépendances temporelles ;
- les changements de régime ;
- le choix de la période d'évaluation.

Un modèle peut réussir un test de couverture moyenne, comme Kupiec, tout en échouant à capturer correctement le regroupement temporel des violations.

## Limites de l'évaluation économique

Les comparaisons de richesse, de Sharpe ou de drawdown dépendent des règles de construction retenues, des hypothèses de friction et du contexte historique considéré.

Ces indicateurs :

- ne remplacent pas une évaluation de robustesse ;
- ne constituent pas une preuve de supériorité future ;
- ne doivent pas être convertis en promesse de performance ;
- doivent être lus comme des résultats historiques ou simulés, pas comme des garanties.

L'évaluation économique sert à tester si les signaux de risque ont une utilité historique dans un protocole contrôlé. Elle ne constitue pas une validation d'une stratégie réelle d'investissement.

## Limite de la stratégie d'allocation dynamique

La stratégie d'allocation dynamique ou de **risk-targeting** utilisée dans le dashboard est une règle expérimentale de backtesting économique. Dans le code actuel, elle sert à évaluer si les prévisions de risque, surtout la VaR prédite, peuvent améliorer historiquement un profil rendement-risque simulé.

Note d'implémentation actuelle : le poids effectif est construit à partir d'un budget roulant de VaR et utilisé avec un retard d'une période. Le HMM ne pilote pas directement le poids de la simulation économique actuelle, sauf modification explicite du code.

Elle ne constitue pas une recommandation d'investissement réelle. Les poids calculés par la stratégie sont des **poids de simulation**, dépendants du modèle, de la période de test, de la période de calibration, du choix des quantiles, de la mesure de risque retenue et des hypothèses de backtesting.

Le chatbot peut expliquer comment les poids simulés réagissent aux indicateurs de risque, mais il ne doit pas dire à l'utilisateur quelle allocation adopter réellement.

Une formulation correcte est :

> Le poids simulé diminue lorsque le risque prévu augmente dans la règle de backtesting économique.

Une formulation à éviter est :

> Il faut réduire l'exposition maintenant.

## Limites de la règle de risk-targeting conditionnelle au régime

Lorsque le budget de risque dépend du régime HMM, la règle devient sensible à la qualité de classification du régime. Une mauvaise identification du régime peut conduire à un budget de risque inadapté.

Dans une règle de type 25 % / 75 %, le budget de risque peut être plus prudent en régime stressé et plus permissif en régime calme :

$$
B(S_t)=
\begin{cases}
Q_{0.25}\left(|\widehat{\operatorname{VaR}}_{\alpha}|\right), & \text{si } S_t \text{ correspond à un régime stressé},\\
Q_{0.75}\left(|\widehat{\operatorname{VaR}}_{\alpha}|\right), & \text{si } S_t \text{ correspond à un régime calme}.
\end{cases}
$$

Le poids simulé peut alors s'écrire :

$$
w_t =
\min\left(
1,
\frac{B(S_t)}
{|\widehat{\operatorname{VaR}}_{\alpha,t+1|t}|}
\right)
$$

Cette règle réduit l'arbitraire d'un budget unique, mais elle introduit d'autres choix de protocole : définition des régimes, choix des quantiles, période de calibration, mesure de risque utilisée et règle de plafonnement.

Les poids produits par cette règle sont des poids simulés pour l'évaluation économique historique. Ils ne doivent pas être interprétés comme des recommandations d'allocation réelle.

## Limites de la reconstruction d'un niveau de prix

Lorsqu'un niveau de prix indicatif est reconstruit à partir d'un rendement prévu, cette transformation a une finalité principalement visuelle et pédagogique.

La relation utilisée est :

$$
\widehat{P}_{t+1|t}
=
P_t \exp(\hat{r}_{t+1|t})
$$

Cette formule convertit un rendement logarithmique prévu en niveau d'indice implicite à partir du dernier niveau observé. Elle ne transforme pas le modèle en véritable modèle de prévision directe des prix.

Cette reconstruction ne capture pas toute la dynamique potentielle d'une trajectoire multi-périodes, ne tient pas compte de tous les scénarios possibles de volatilité, et ne doit pas être présentée comme un objectif de marché ou comme un niveau garanti.

## Limites liées aux données de marché marocaines

Comme pour toute application sur un marché donné, la qualité de l'interprétation dépend :

- de la qualité et de la continuité des données ;
- de la liquidité relative du marché ;
- des ruptures institutionnelles, macroéconomiques ou sectorielles ;
- des épisodes rares qui peuvent être sous-représentés dans l'historique disponible ;
- des effets éventuels de calendrier, de cotation ou de disponibilité des données.

Le contexte marocain doit donc être traité avec rigueur terminologique et prudence empirique.

## Absence de garantie hors échantillon

Aucun modèle quantitatif de risque ne garantit sa validité hors échantillon. Les changements de politique économique, de structure de marché, de volatilité internationale ou de comportement des investisseurs peuvent déformer rapidement les régularités estimées dans le passé.

Une bonne performance historique ou un backtest satisfaisant ne signifie donc pas que le modèle restera fiable dans toutes les conditions futures.

## Limites de la couche RAG

La couche RAG repose sur deux familles d'information :

- des documents fixes de connaissance, comme ceux du présent dossier ;
- un contexte dynamique alimenté par les sorties effectives du dashboard.

Le chatbot peut fournir une explication rigoureuse seulement si :

- la métrique demandée existe dans le contexte dynamique ;
- la terminologie employée est conforme à la méthodologie ;
- aucune extrapolation non fondée n'est introduite ;
- les documents récupérés correspondent réellement à la question posée.

En l'absence de donnée dynamique, le chatbot doit expliquer le concept, mais ne pas inventer de valeur.

## Limites de l'analyse financière descriptive

Le chatbot peut produire une analyse financière descriptive du risque, par exemple en expliquant qu'une combinaison de volatilité élevée, de VaR plus négative et d'ES plus sévère indique un environnement de risque plus tendu.

Cependant, cette analyse reste :

- conditionnelle aux sorties du modèle ;
- dépendante des données disponibles ;
- descriptive et non prescriptive ;
- limitée à l'interprétation du risque ;
- non assimilable à une recommandation d'allocation réelle.

Elle ne doit jamais être transformée en recommandation d'investissement, en conseil de trading ou en instruction d'allocation réelle.

## Données manquantes ou indisponibles

Si une valeur n'est pas présente dans le contexte dynamique du dashboard, le chatbot doit le signaler explicitement. Il ne doit pas reconstruire, deviner ou approximer une valeur absente.

Formulation recommandée :

> Cette information n'est pas disponible dans le contexte fourni par le dashboard.

Le chatbot peut ensuite expliquer le concept général, mais sans inventer de chiffre.

## Règle générale de prudence

Toute réponse doit rappeler implicitement ou explicitement que le système est un **outil d'interprétation du risque**, et non un système de décision autonome en investissement.

Les résultats doivent être présentés comme des sorties modélisées, conditionnelles, dépendantes des données et soumises à incertitude.
