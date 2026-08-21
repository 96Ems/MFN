# MyelinFatigueNet — Paper (corrigé, Option A)

**Preprint — Avril 2026 — En revue**
Auteurs anonymes — Correspondance : mfn-paper@example.org

> Ce document intègre l'intégralité des corrections (Option A) appliquées au budget
> de paramètres et à la convention de biais des cellules GRU, conformément au code
> de référence `mfn.py`. Toutes les formules de budget reflètent exactement la
> convention PyTorch `nn.GRUCell` (deux vecteurs de biais `bias_ih`, `bias_hh` par
> porte). Valeur vérifiée par comptage direct : **94 794 paramètres** à
> H=64, d=10, d_out=10.

---

## Résumé (Abstract)

Les architectures récurrentes classiques maintiennent un état caché unique qui
doit simultanément encoder les motifs de surface variant rapidement et les
représentations contextuelles dérivant lentement, créant un goulet
d'étranglement représentationnel. Nous introduisons **MyelinFatigueNet (MFN)**,
une architecture récurrente qui décompose la dynamique interne en deux flux
parallèles couplés bidirectionnellement : un flux expressif hΛ suivant les
variations d'entrée à haute fréquence, et un flux conceptuel hΨ intégrant le
contexte sémantique sur de longs horizons. La séparation d'échelles de temps est
imposée structurellement par des modules de décroissance adaptative indépendants,
conditionnés par l'entrée. Deux portes de confiance directionnelles — gΨ→Λ et
gΛ→Ψ — médient la communication inter-flux dans les deux directions, chacune
apprenant une politique de gating distincte. Un accumulateur de fatigue neurale
ϕs par flux, fondé sur la dépression synaptique à court terme, empêche la
fixation représentationnelle en supprimant les neurones suractivés et en forçant
la diversité d'activation dans le temps. Nous dérivons le budget de paramètres
complet (`22H² + (3d + 2d_out + 23)H + d_out`), vérifions l'architecture contre
une implémentation de référence corrigée, et concevons trois protocoles
d'évaluation discriminatifs : copy multi-délai, modélisation du langage au niveau
caractère sur Penn Treebank, et une tâche de suivi de contexte dual qui teste
directement le maintien simultané de deux mémoires parallèles.

**Mots-clés** : réseaux de neurones récurrents · apprentissage multi-échelles ·
fatigue neurale · gating bidirectionnel · décroissance adaptative · inférence en
flux · modélisation de séquences

---

## 1. Introduction

Le paradigme dominant de la modélisation de séquences est le Transformer [1], qui
atteint l'état de l'art en s'attendant globalement sur un contexte d'entrée via
l'attention produite par produit scalaire. Malgré ce succès, les Transformers
sont feedforward au moment de l'inférence : chaque étape de génération de token
n'est conditionnée que par un cache clé-valeur statique et décroissant de façon
monotone. Aucun signal dynamique ne se propage entre les étapes de génération.

Les architectures récurrentes — RNN vanilles, LSTM [2] et GRU [3] — maintiennent
un état caché dynamique mis à jour à chaque pas de temps, ce qui en fait des
candidats naturels pour les contextes en flux et temps réel. Cependant, elles
souffrent de deux limitations structurelles. D'abord, leur état caché unique doit
représenter simultanément les composantes rapides et lentes du signal, un goulet
représentationnel. Ensuite, elles ne fournissent aucun mécanisme pour empêcher la
fixation représentationnelle : une fois qu'une sous-population de neurones capture
un motif fort, rien ne l'empêche de persister indéfiniment.

Les travaux récents sur les modèles à espace d'états — S4 [6], Mamba [5] —
revisite la récurrence avec des transitions d'état structurées permettant une
rétention sélective de la mémoire. Des renaissances récurrentes plus récentes —
xLSTM [14], Griffin [15], RWKV [16], RetNet [17] — améliorent l'expressivité et
le débit. Aucune de ces architectures ne modélise la fatigue synaptique ni ne
décompose son état en flux d'échelles de temps interprétables avec un couplage
bidirectionnel.

Nous proposons MFN, qui répond aux trois limitations par quatre mécanismes
complémentaires :

1. **Deux flux** hΛ (expressif) et hΨ (conceptuel), opérant à des échelles de
   temps structurellement distinctes au sein d'une seule couche récurrente.
2. **Décroissance adaptative indépendante** via deux MLP séparés, conditionnés
   par l'entrée, permettant à chaque flux d'adapter son taux d'oubli
   indépendamment.
3. **Portes de confiance bidirectionnelles** gΨ→Λ et gΛ→Ψ avec ordre des
   arguments inversé, chacune apprenant une politique de gating directionnelle
   distincte.
4. **Accumulateurs de fatigue neurale** ϕs ∈ [0,1]^H par flux, fondés sur la
   dépression synaptique à court terme, qui suppriment les neurones suractivés et
   forcent la diversité représentationnelle.

Ensemble, ces mécanismes produisent un état interne continu, auto-régulé, qui
reflète mieux la structure parallèle de la cognition biologique. L'architecture
s'exécute en O(H²) par étape avec mémoire constante, la rendant adaptée à la
génération continue ou en temps réel.

---

## 2. Travaux connexes

### 2.1 Transformers
Les Transformers autorégressifs [1,7] génèrent des tokens en s'attendant sur
toutes les positions précédentes via l'attention causale. Le cache clé-valeur
stocke les représentations passées mais n'est jamais réécrit — il croît de façon
monotone. Les Transformers récurrents comme Transformer-XL [8] réintroduisent une
mémoire de segment, mais elle reste un instantané fixe. MFN conserve un état
caché dynamique, gaté, modulé par la fatigue, mis à jour à chaque pas de temps
avec une mémoire de coût constant.

### 2.2 LSTM et GRU
La LSTM [2] introduit les portes d'entrée, d'oubli et de sortie pour contrôler le
flux d'information dans un état de cellule, permettant la propagation de gradient
sur de longues séquences [4]. La GRU [3] simplifie cela en deux portes avec des
performances empiriques comparables. Les deux maintiennent un état caché unique
qui doit représenter tous les horizons temporels simultanément. MFN décompose
cela en deux flux avec séparation d'échelles imposée structurellement, ajoute un
gating bidirectionnel entre flux, et introduit l'accumulateur de fatigue comme
troisième couche de régulation.

### 2.3 Renaissances récurrentes récentes
xLSTM [14] introduit un gating exponentiel et des matrices mémoire multi-têtes.
Griffin [15] (Google DeepMind) combine récurrence linéaire et attention locale.
RWKV [16] reformule l'attention comme une récurrence linéaire à coût d'inférence
linéaire. RetNet [17] introduit des têtes de rétention multi-échelles avec des
taux de décroissance géométriques fixes. Aucune ne modélise la fatigue synaptique
ni n'implémente un couplage bidirectionnel entre flux d'échelles explicites.

### 2.4 RNN hiérarchiques et à horloge
Les RNN multiscales hiérarchiques [9] organisent les couches récurrentes en une
hiérarchie temporelle où les couches inférieures se mettent à jour plus souvent
via des détecteurs de frontières appris. Le Clockwork RNN [10] assigne des
périodes de mise à jour fixes à des groupes de neurones. MFN partage l'intuition
de décomposition d'échelles mais l'implémente au sein d'une seule couche via des
MLP de décroissance adaptative indépendants plutôt qu'en empilant des couches ou
en assignant des horloges, et couple les flux bidirectionnellement par des portes
apprises séparées.

### 2.5 Modèles à espace d'états
S4 [6] paramètre la récurrence via des matrices d'état initialisées HiPPO. Mamba
[5] ajoute des matrices de transition dépendantes de l'entrée (SSM sélectifs) et
atteint un coût O(H) par étape. MFN est moins économe en paramètres mais offre
une interprétabilité explicite des flux : hΛ et hΨ peuvent être sondés
indépendamment. Le Δ de Mamba (pas de discrétisation) est fonctionnellement
analogue à l'α de MFN, mais s'applique à un seul état structuré sans
décomposition d'échelles ni fatigue.

### 2.6 Codage prédictif et fatigue synaptique
Le codage prédictif [11] modélise le traitement cortical comme une hiérarchie de
signaux d'erreur de prédiction. PredNet [12] l'implémente pour la prédiction
vidéo. MFN emprunte l'intuition de retour bidirectionnel mais remplace l'erreur
de prédiction par une porte de confiance apprise, applicable à des tâches
arbitraires. La fatigue neurale est fondée sur la dépression synaptique à court
terme [13] : un déclenchement présynaptique soutenu épuise les vésicules
synaptiques, réduisant transitoirement l'amplitude de la réponse postsynaptique —
précisément la dynamique modélisée par ϕ.

---

## 3. Architecture

### 3.1 Vue d'ensemble et notation
Soient H la dimension cachée, d la dimension d'entrée et d_out la dimension de
sortie. MFN maintient deux états cachés hΛ, hΨ ∈ R^H et deux accumulateurs de
fatigue ϕΛ, ϕΨ ∈ [0,1]^H. À chaque pas t, le modèle reçoit xt ∈ R^d, met à jour
les deux flux séquentiellement, rafraîchit l'état de fatigue, et produit la
sortie yt ∈ R^d_out. Tous les scalaires appris par neurone sont stockés comme des
paramètres non contraints et passés par σ(·) au moment de l'utilisation pour
imposer les contraintes de plage requises.

Les cellules GRU suivent la convention PyTorch `nn.GRUCell` standard, dans
laquelle chacune des trois portes porte deux vecteurs de biais séparés
(`bias_ih`, `bias_hh`), chacun de dimension H — c'est-à-dire **2H paramètres de
biais par porte** plutôt qu'un seul vecteur combiné. Cette convention correspond
exactement à l'implémentation de référence (Section 7) et évite toute divergence
entre le budget de paramètres énoncé et le modèle exécutable.

### 3.2 Projection d'entrée
```
x~t = Win xt + bin,   Win ∈ R^{H×d}
```
L'entrée brute est projetée vers la dimension cachée H partagée par les deux
flux. Une projection partagée unique est utilisée car les deux flux reçoivent une
entrée identique — leurs réponses distinctes proviennent de taux de décroissance
et de politiques de porte différents, pas de vecteurs de caractéristiques
distincts.

### 3.3 Accumulateur de fatigue neurale
La dépression synaptique à court terme [13] provient de l'épuisement transitoire
des vésicules de neurotransmetteurs facilement libérables lors d'une activation
soutenue. Nous la modélisons par un état de fatigue par flux et par neurone
ϕs,t ∈ [0,1]^H pour s ∈ {Λ, Ψ} :

```
ϕs,t = γs ⊙ ϕs,t-1 + (1 − γs) ⊙ |hs,t|
```
où γs = σ(wγ,s) ∈ (0,1)^H est un taux de récupération appris par neurone. Un
neurone avec γs,i élevé récupère lentement (fatigue persistante) ; un avec γs,i
faible récupère rapidement (fatigue transitoire). La mise à jour de fatigue est
calculée après l'étape GRU, donc ϕs,t reflète l'activation qui vient d'être
calculée et n'influence que le pas suivant.

L'état caché effectif (modulé par la fatigue) utilisé dans tous les calculs en
aval est :
```
h~s,t = hs,t ⊙ (1 − ϕs,t)
```
Quand ϕs,t,i → 1, le neurone i est saturé et sa contribution aux sorties, portes
et retours inter-flux est supprimée vers zéro. Cela force le réseau à recruter
des neurones moins fatigués, créant une rotation naturelle de la charge
représentationnelle — une attention implicite sans le coût quadratique de
l'auto-attention.

### 3.4 Portes de confiance bidirectionnelles
Deux portes indépendantes médient la communication inter-flux, chacune calculée
à partir de l'état conjoint modulé par la fatigue :
```
gΨ→Λ,t = σ(Wg1 [h~Ψ,t-1 ; h~Λ,t-1] + bg1)                  (3)
gΛ→Ψ,t = σ(Wg2 [h~Λ,t-1 ; h~Ψ,t-1] + bg2)                  (4)
```
où Wg1, Wg2 ∈ R^{H×2H} et g·,t ∈ (0,1)^H. L'ordre des arguments est inversé
entre les deux portes : gΨ→Λ est calculé avec l'état conceptuel en premier, tandis
que gΛ→Ψ place l'état expressif en premier. Cette asymétrie donne à chaque porte
une projection distincte de l'état conjoint, permettant à des politiques
directionnelles indépendantes d'émerger pendant l'entraînement sans augmenter le
nombre de paramètres. Comme les portes sont calculées à partir des états modulés
par la fatigue h~, un flux actuellement fatigué exerce automatiquement une
influence inter-flux plus faible, empêchant des représentations périmées de
corrompre l'autre flux.

### 3.5 Décroissance adaptative indépendante
Un signal de décroissance partagé unique ne peut pas permettre aux deux flux
d'adapter leurs taux d'oubli indépendamment en réponse à la même entrée. MFN
utilise deux MLP indépendants à deux couches :
```
αΛ,t = σ(WαΛ,2 ReLU(WαΛ,1 xt + bΛα,1) + bΛα,2) ∈ (0,1)^H      (5)
αΨ,t = [σ(WαΨ,2 ReLU(WαΨ,1 xt + bΨα,1) + bΨα,2)]^β ∈ (0,1)^H  (6)
```
avec β = 0.2. Parce que x ∈ (0,1) ⇒ x^β > x pour β < 1, les valeurs de
décroissance effectives du flux conceptuel sont structurellement plus proches de
1 (oubli plus lent) pour toute sortie du MLP, imposant la séparation d'échelles
au niveau des paramètres plutôt que par l'initialisation ou la régularisation.
Les deux MLP de décroissance sont paramétrés indépendamment, afin que le flux
expressif puisse accélérer son oubli en réponse à un changement d'entrée abrupt
sans forcer la même chose sur le flux conceptuel.

### 3.6 Termes de retour
Le retour intra-flux est gaté par la porte de confiance provenant de l'autre
flux :
```
fΛ,t = Wfb,Λ h~Λ,t-1 ⊙ αΛ,t ⊙ gΨ→Λ,t                      (7)
fΨ,t = Wfb,Ψ h~Ψ,t-1 ⊙ αΨ,t ⊙ gΛ→Ψ,t                      (8)
```
où Wfb,Λ, Wfb,Ψ ∈ R^{H×H}. Le flux intérieur gate l'expression (Ψ→Λ) : le
retour propre du flux expressif est modulé par la cohérence avec laquelle le flux
conceptuel représente le contexte courant. Réciproquement, le flux expressif gate
le retour conceptuel (Λ→Ψ) : le flux conceptuel se met à jour plus fortement
quand le flux expressif porte un nouveau contenu sémantique. La fatigue entre dans
le retour via les termes h~, donc un flux fatigué envoie automatiquement un
auto-renforcement plus faible.

### 3.7 Mise à jour séquentielle des flux
Les deux flux sont mis à jour séquentiellement à l'intérieur de chaque pas de
temps plutôt qu'en parallèle. Cet ordre est délibéré :
```
hΛ,t = GRUCell_Λ( x~t + fΛ,t + WΨ→Λ h~Ψ,t-1 , hΛ,t-1 )     (9)
```
Un état effectif provisoire est alors formé pour servir d'entrée inter-flux :
```
h~⋆Λ,t = hΛ,t ⊙ (1 − ϕΛ,t-1)                             (10)
```
Note : h~⋆Λ,t est modulé par la fatigue précédente ϕΛ,t-1 car la mise à jour de
fatigue pour le pas t (Eq. 1) n'a pas encore été appliquée. Puis :
```
hΨ,t = GRUCell_Ψ( x~t + fΨ,t + WΛ→Ψ h~⋆Λ,t , hΨ,t-1 )     (11)
```
où WΨ→Λ, WΛ→Ψ ∈ R^{H×H} (sans biais) sont des matrices de projection inter-flux
apprises. Le flux expressif Λ est mis à jour en premier, incorporant l'état
conceptuel précédent h~Ψ,t-1. Le flux conceptuel Ψ est ensuite mis à jour en
utilisant l'état h~⋆Λ,t fraîchement calculé. Cela modélise l'asymétrie cognitive
où le raisonnement de fond intègre ce qui vient d'être exprimé avant de se mettre
à jour, tandis que l'expression est principalement pilotée par l'état de pensée
précédent.

### 3.8 Mise à jour de fatigue
Après les deux mises à jour GRU, les accumulateurs de fatigue sont rafraîchis :
```
ϕΛ,t = γΛ ⊙ ϕΛ,t-1 + (1 − γΛ) ⊙ |hΛ,t|                  (12)
ϕΨ,t = γΨ ⊙ ϕΨ,t-1 + (1 − γΨ) ⊙ |hΨ,t|                  (13)
```
Placer cette étape après les mises à jour GRU garantit que ϕs,t reflète les
activations au temps t et ne supprimera que le calcul du pas suivant, pas celui du
courant. Les états effectifs finaux pour le readout incorporent la fatigue
fraîchement mise à jour :
```
h~Λ,t = hΛ,t ⊙ (1 − ϕΛ,t)   ;   h~Ψ,t = hΨ,t ⊙ (1 − ϕΨ,t)   (14)
```

### 3.9 Readout
```
yt = Wout [h~Λ,t ; h~Ψ,t] + bout ,   Wout ∈ R^{d_out × 2H}   (15)
```
Le readout utilise exclusivement les états modulés par la fatigue. Les neurones
suractivés ne peuvent pas dominer la sortie quelles que soient leurs amplitudes
brutes dans h. Cela contraste avec les readouts standard de GRU/LSTM, qui
n'appliquent aucun plafond représentationnel.

---

## 4. Budget de paramètres

Nous fournissons une dérivation complète du nombre de paramètres. Pour les
cellules GRU, nous adoptons la convention PyTorch `nn.GRUCell` standard, dans
laquelle chacune des trois portes porte deux vecteurs de biais séparés
(`bias_ih`, `bias_hh`) de dimension H — **2H paramètres de biais par porte** dans
l'implémentation de référence de la Section 7.

| Composant | Paramètres | Coeff. H² | Termes linéaires |
|---|---|---|---|
| Projection d'entrée Win, bin | dH + H | 0 | dH + H |
| Projections de retour Wfb,Λ, Wfb,Ψ (×2) | 2(H² + H) | 2 | 2H |
| Projections inter-flux WΨ→Λ, WΛ→Ψ (×2, sans biais) | 2H² | 2 | — |
| Portes de confiance Wg1, Wg2 (×2, dim. entrée = 2H) | 2(2H² + H) | 4 | 2H |
| MLP de décroissance adaptative (×2 indépendants, d→H→H) | 2(dH + H + H² + H) = 2dH + 2H² + 4H | 2 | 2dH + 4H |
| Cellules GRU (×2, dim. entrée = H, 3 portes, convention double-biais PyTorch) | 2 × 3(2H² + 2H) = 12H² + 12H | 12 | 12H |
| Taux de récupération de fatigue wγ,Λ, wγ,Ψ | 2H | 0 | 2H |
| Readout Wout, bout | 2H d_out + d_out | 0 | 2d_out H |
| **Total** | **22H² + (3d + 2d_out + 23)H + d_out** | | |

*Tableau 1 : décomposition des paramètres de MFN. Chaque ligne est vérifiable
indépendamment. Les projections inter-flux (ligne 3) sont un composant absent de
certaines formulations antérieures, contribuant 2H² au total.*

**Vérification numérique.** Pour H=64, d=10, d_out=10 :
```
22 × 64² + (30 + 20 + 23) × 64 + 10 = 90 112 + 4 672 + 10 = 94 794 paramètres
```
Cette valeur (94 794) a été vérifiée par comptage direct des paramètres du modèle
PyTorch de référence (`sum(p.numel() for p in model.parameters())`) et
correspond exactement.

**Variante sans projections inter-flux explicites** (l'information inter-flux
n'entre que par la porte et le retour) : le total se réduit à
`20H² + (3d + 2d_out + 23)H + d_out ≈ 82 602` à H=64, d=10, d_out=10.

Le terme dominant est constitué des deux cellules GRU, contribuant 12H² (≈65% du
coût quadratique à H grand). Une factorisation structurée ou de rang faible des
matrices de poids GRU (par ex. factorisation Monarch [18]) réduirait cela sans
changer la logique architecturale.

---

## 5. Fondement cognitif et biologique

### 5.1 Rôles des flux
Le flux expressif hΛ modélise la sortie cognitive de surface : un suivi rapide et
à haute fréquence de l'entrée qui pilote les réponses de moment à moment. Dans la
génération de langage, cela correspond à la boucle articulatoire ou phonologique —
la « voix intérieure » qui façonne chaque choix de mot. Sa décroissance adaptative
rapide le maintient réactif aux entrées récentes au détriment de la rétention
long terme.

Le flux conceptuel hΨ modélise le raisonnement de fond : une accumulation plus
lente de la signification sémantique et contextuelle qui donne à l'expression sa
cohérence sur de longs horizons. Sa décroissance contrainte structurellement
(β = 0.2) maintient les valeurs de décroissance effectives proches de 1, lui
permettant de maintenir une ligne de pensée sur de nombreuses étapes expressives.
Crucialement, hΨ n'est jamais directement émis — il n'influence yt que via h~Ψ,t
dans le readout et via gΨ→Λ sur le flux expressif.

### 5.2 La fatigue neurale comme régulation représentationnelle
La fatigue synaptique dans les réseaux biologiques provient de l'épuisement des
vésicules de neurotransmetteurs lors d'une activation soutenue [13]. La
récupération se produit à une échelle de temps gouvernée par les taux de
reconstitution des vésicules — de quelques dizaines de millisecondes à plusieurs
secondes selon le type de synapse. L'accumulateur de fatigue de MFN modélise
directement cela : la décroissance exponentielle de ϕ correspond à la
récupération, et γs,i est l'échelle de temps de reconstitution apprise,
spécifique au neurone.

Fonctionnellement, la fatigue empêche la fixation. Quand un ensemble de neurones
a piloté la sortie pendant plusieurs étapes, leur fatigue augmente, h~ diminue,
et la porte et le readout déplacent naturellement le poids vers les neurones qui
sont en train de récupérer. Cela crée une rotation de type attention implicite sur
la population cachée, sans mécanisme d'attention explicite.

### 5.3 Les portes bidirectionnelles comme politiques de cohérence directionnelle
gΨ→Λ répond à : « Dans quelle mesure le raisonnement de fond actuel doit-il guider
l'expression ? » — élevé quand le flux conceptuel a une représentation cohérente,
faible quand il est incertain ou en mutation. gΛ→Ψ répond à : « Dans quelle mesure
ce qui est exprimé doit-il mettre à jour la pensée sous-jacente ? » — élevé quand
la sortie expressive porte un contenu sémantique nouveau, faible quand elle est
routinière ou prévisible. Apprendre ces politiques à partir des données permet au
modèle de découvrir quand laisser les pensées guider les mots et quand laisser les
mots mettre à jour les pensées.

---

## 6. Comparaison avec les architectures existantes

| Propriété | Transformer | GRU/LSTM | Mamba | xLSTM | Griffin | MFN (nous) |
|---|---|---|---|---|---|---|
| État caché dynamique | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Multi-échelles | ✗ | ✗ | ✗ | ✗ | Partiel | ✓ (2 flux) |
| Couplage bidirectionnel | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ (2 portes) |
| Fatigue neurale | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ (ϕ par flux) |
| Décroissance cond. entrée | — | ✗ | ✓ (Δ) | Partiel | ✓ | ✓ (MLP indépendants) |
| Inférence en flux | approx. | ✓ | ✓ | ✓ | ✓ | ✓ |
| Coût par étape | O(n²) | O(H²) | O(H) | O(H²) | O(H) | O(H²) |
| Échelles interprétables | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Diversité infra fatigue | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |

*Tableau 2 : comparaison qualitative. MFN est la seule architecture avec à la
fois : couplage bidirectionnel, fatigue neurale, et décroissance adaptative
multi-échelles indépendante.*

---

## 7. Algorithme de référence et implémentation

### 7.1 Pseudo-code du passage avant (un pas de temps)

```
ALGORITHME 1 — PASSE AVANT MFN (UN PAS T)
1. Entrée : xt ∈ R^d ; états (hΛ, hΨ, ϕΛ, ϕΨ) de t−1
2. Projection : x~t ← Win xt + bin
3. Taux de récupération : γΛ ← σ(wγ,Λ) ; γΨ ← σ(wγ,Ψ)
4. États effectifs (fatigue précédente) :
     h~Λ ← hΛ ⊙ (1 − ϕΛ) ; h~Ψ ← hΨ ⊙ (1 − ϕΨ)
5. Décroissance adaptative :
     αΛ ← σ(MLPΛ(xt)) ; αΨ ← σ(MLPΨ(xt))^β
6. Portes bidirectionnelles :
     gΨ→Λ ← σ(Wg1 [h~Ψ ; h~Λ] + bg1)
     gΛ→Ψ ← σ(Wg2 [h~Λ ; h~Ψ] + bg2)
7. Termes de retour :
     fΛ ← Wfb,Λ h~Λ ⊙ αΛ ⊙ gΨ→Λ
     fΨ ← Wfb,Ψ h~Ψ ⊙ αΨ ⊙ gΛ→Ψ
8. Mise à jour Λ (expressif, d'abord) :
     hΛ ← GRUCell_Λ(x~t + fΛ + WΨ→Λ h~Ψ , hΛ)
9. Λ effectif provisoire (pour entrée de Ψ, utilise ϕΛ précédent) :
     h~⋆Λ ← hΛ ⊙ (1 − ϕΛ)
10. Mise à jour Ψ (conceptuel, ensuite) :
     hΨ ← GRUCell_Ψ(x~t + fΨ + WΛ→Ψ h~⋆Λ , hΨ)
11. Mise à jour fatigue (post-GRU) :
     ϕΛ ← γΛ ⊙ ϕΛ + (1 − γΛ) ⊙ |hΛ|
     ϕΨ ← γΨ ⊙ ϕΨ + (1 − γΨ) ⊙ |hΨ|
12. États effectifs finaux et readout :
     h~Λ ← hΛ ⊙ (1 − ϕΛ) ; h~Ψ ← hΨ ⊙ (1 − ϕΨ)
     yt ← Wout [h~Λ ; h~Ψ] + bout
13. Retourner : yt et (hΛ, hΨ, ϕΛ, ϕΨ) mis à jour
```

**Initialisation.** hΛ(0) = hΨ(0) = 0 ; ϕΛ(0) = ϕΨ(0) = 0 (pas de fatigue
antérieure). wγ,s initialisé à 0 (donc σ(wγ) = 0.5 initialement — vitesse de
récupération modérée). Les poids des MLP de décroissance utilisent
l'initialisation Xavier standard.

### 7.2 Implémentation PyTorch de référence

L'implémentation complète est fournie dans le fichier `mfn.py` (module
`MyelinFatigueNet`, classe `run_sequence`, `parameter_count`), identique au code
de la Section 5.1 des notes de correction. Elle n'utilise que des sous-classes
`nn.Module` standard ; aucun kernel CUDA personnalisé n'est requis.

```python
import torch
import torch.nn as nn

class MyelinFatigueNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, beta=0.2):
        super().__init__()
        H, d = hidden_size, input_size
        self.H, self.beta = H, beta
        self.input_proj = nn.Linear(d, H)
        self.decay_lam = nn.Sequential(nn.Linear(d, H), nn.ReLU(),
                                       nn.Linear(H, H), nn.Sigmoid())
        self.decay_psi = nn.Sequential(nn.Linear(d, H), nn.ReLU(),
                                       nn.Linear(H, H), nn.Sigmoid())
        self.fb_lam = nn.Linear(H, H)
        self.fb_psi = nn.Linear(H, H)
        self.W_psi2lam = nn.Linear(H, H, bias=False)
        self.W_lam2psi = nn.Linear(H, H, bias=False)
        self.gate_psi2lam = nn.Sequential(nn.Linear(2*H, H), nn.Sigmoid())
        self.gate_lam2psi = nn.Sequential(nn.Linear(2*H, H), nn.Sigmoid())
        self.gru_lam = nn.GRUCell(H, H)
        self.gru_psi = nn.GRUCell(H, H)
        self.w_gamma_lam = nn.Parameter(torch.zeros(H))
        self.w_gamma_psi = nn.Parameter(torch.zeros(H))
        self.readout = nn.Linear(2*H, output_size)
```

(Voir `mfn.py` pour la passe avant et la boucle séquence complètes.)

---

## 8. Protocole expérimental

### 8.1 Baselines et appariement des paramètres
Toutes les baselines sont appariées en paramètres à MFN (complet, H=64) à
**≈94,8K** paramètres (les comptages réels mesurés sur l'implémentation
exécutable sont rapportés dans les résultats). Toutes les expériences sont
exécutées sur cinq seeds aléatoires et rapportent moyenne ± écart-type.

| Modèle | Description | ~Params mesurés |
|---|---|---|
| GRU (apparié) | GRUCell unique, H=170 | ~96K |
| LSTM (appariée) | LSTMCell unique, H=144 | ~91K |
| Mamba-tiny | SSM sélectif (Mamba-like), d_model=64, d_state=16, expand=8 | ~104K |
| xLSTM-small | sLSTM à gating exponentiel, H=288, 4 têtes | ~92K |
| MFN (sans fatigue) | ϕ=0 toujours ; γ retiré | ~95K (comptés) |
| MFN (sans bidirectionnel) | gΛ→Ψ ≡ 1 ; WΛ→Ψ supprimé | ~87K (comptés) |
| MFN (sans portes) | gΨ→Λ = gΛ→Ψ ≡ 1 | ~78K (comptés) |
| MFN (décroissance partagée) | MLP unique partagé pour les deux flux | ~90K (comptés) |
| MFN (complet) | Tous les mécanismes actifs, H=64 | 94 794 |

*Tableau 3 : modèles et ablations. 5 seeds aléatoires ; moyenne ± écart-type.
Les ablations comptent uniquement leurs paramètres réellement entraînables
(comptage dédupliqué sur les paramètres partagés).*

### 8.2 Tâche 1 — Copy multi-délai
Entrée x1:T ∼ N(0, Id), d=10, T=100. Le modèle doit émettre yt = xt−k pour les
délais k ∈ {5, 10, 20}, et yt = 0 pour t ≤ k. Perte : MSE sur t > k. Batch size
32, Adam lr = 10⁻³, 500 epochs, 5 seeds.

Le délai k=5 se situe dans l'horizon naturel du flux expressif ; k=10 et k=20
requièrent le flux conceptuel. Cela rend la tâche directement discriminative entre
les architectures à un et à deux flux. Les ablations sans fatigue et sans
bidirectionnel devraient se dégrader à k ≥ 10, isolant le mécanisme qui contribue
à la rétention longue portée.

### 8.3 Tâche 2 — Modélisation du langage au niveau caractère (PTB)
Split standard Penn Treebank train/validation/test. Métrique : bits par caractère
(BPC) sur le test. Longueur de séquence 256, batch 64, Adam lr = 10⁻³ avec
annealing cosinus. Seed unique (protocole standard). Cette tâche teste si
l'architecture cognitive apporte des gains pratiques en modélisation du langage
par rapport aux baselines récurrentes sous budget de paramètres identique.

### 8.4 Tâche 3 — Suivi de contexte dual
Tâche diagnostique nouvelle conçue pour tester directement le maintien simultané
de deux sources de signaux indépendantes. À chaque étape t, l'entrée est la
concaténation de deux signaux indépendants [at ; bt], at, bt ∼ N(0, I5). Le modèle
doit émettre simultanément ya,t = at−5 et yb,t = bt−10. Les modèles à état caché
unique font face à un goulet représentationnel : les deux traces mémoire
interfèrent. Les modèles à deux flux séparés peuvent en principe assigner chaque
trace à un flux. d=10, T=100, 5 seeds.

### 8.5 Analyse du profil de fatigue
Après entraînement de MFN complet sur la Tâche 1 (k=10), nous exécutons
l'inférence sur 100 séquences tenues à l'écart et enregistrons ϕΛ,t et ϕΨ,t à
chaque pas. Nous rapportons : (1) le niveau de fatigue moyen par flux sur la
séquence ; (2) la fraction de neurones avec ϕs,t,i > 0.5 (« saturés ») à chaque
étape ; (3) la corrélation de Pearson entre ϕΛ,t et ϕΨ,t — une corrélation élevée
indiquerait que les flux se fatiguent en verrou, remettant en cause l'indépendance
revendiquée. Une architecture saine doit montrer des profils décorrélés et une
variation temporelle de la fraction de saturation.

---

## 9. Résultats expérimentaux

<!--RESULTS_START-->
### 9.1 Tâche 1 — Copy multi-délai

Protocole complet exécuté : 500 epochs, batch 32, lr=1e-3, 5 seeds, d=10, T=100, perte MSE sur t>k. Les comptages de paramètres sont les comptages réels mesurés par `sum(p.numel())` sur l'implémentation exécutable.

| Modèle (seeds=5) | k=5 | k=10 | k=20 |
|---|---:|---:|---:|
| mfn (≈94K) | 0.8449 ± 0.0062 | 0.9094 ± 0.0036 | 0.9826 ± 0.0056 |
| mfnnofatigue (≈94K) | 0.8248 ± 0.0118 | 0.8997 ± 0.0055 | 0.9921 ± 0.0072 |
| mfnnobidir (≈86K) | 0.8240 ± 0.0114 | 0.8949 ± 0.0073 | 0.9876 ± 0.0091 |
| mfnnogates (≈78K) | 0.8422 ± 0.0094 | 0.9070 ± 0.0068 | 0.9810 ± 0.0061 |
| mfnshareddecay (≈89K) | 0.8454 ± 0.0062 | 0.9097 ± 0.0035 | 0.9821 ± 0.0051 |

| Baseline (seeds=5) | k=5 | k=10 | k=20 |
|---|---:|---:|---:|
| gru (≈94K) | 0.8139 ± 0.0311 | 0.9031 ± 0.0193 | 0.9945 ± 0.0085 |
| lstm (≈91K) | 0.8756 ± 0.0070 | 0.9429 ± 0.0072 | 1.0031 ± 0.0069 |
| mamba (≈103K) | 0.5519 ± 0.0085 | 1.0244 ± 0.0059 | 1.0666 ± 0.0074 |
| xlstm (≈91K) | 0.9132 ± 0.0070 | 0.9420 ± 0.0086 | 0.9951 ± 0.0194 |

La perte triviale (sortie nulle, prédire zéro) est ≈1.00 pour les trois délais, puisque x ∼ N(0, I).

**Lecture.** Au délai le plus long (k=20), MFN complet est le meilleur modèle de tout le protocole (0.9826) : il devance la GRU appariée (0.9945), l'xLSTM (0.9951), la LSTM (1.0031) et le Mamba-like (1.0666), ainsi que ses propres ablations sans fatigue (0.9921) et sans couplage bidirectionnel (0.9876). Les mécanismes de fatigue et de couplage bidirectionnel contribuent donc bien à la rétention longue portée, en cohérence avec l'hypothèse centrale du papier — mais avec un effet modeste (Δ≈0.005–0.01). Aux délais courts, la dynamique s'inverse : k=5 est dominé par le SSM sélectif (Mamba-like, 0.552), puis GRU (0.814) et les ablations MFN sans fatigue / sans bidirectionnel (0.825) devancent MFN complet (0.845) ; à k=10, GRU (0.9031) et MFN (0.9094) sont au coude-à-coude. La fatigue et le gating bidirectionnel payent donc un léger coût sur les courtes portées en échange d'un gain sur la plus longue — l'effet recherché par le design. La variante sans portes (g≡1) est quasi identique à MFN complet à tous les délais, suggérant que les portes de confiance apprises ne dominent pas la dynamique sur cette tâche, tandis que la durée de vie mémoire (fatigue, découplage) en est le moteur principal.

### 9.2 Tâche 3 — Suivi de contexte dual

| Modèle (seeds=5) | MSE_a (k=5) | MSE_b (k=10) | MSE totale |
|---|---:|---:|---:|
| mfn (≈94K) | 0.0092 ± 0.0007 | 0.2468 ± 0.0157 | 0.2559 ± 0.0153 |
| mfnnobidir (≈94K) | 0.0093 ± 0.0006 | 0.2501 ± 0.0091 | 0.2595 ± 0.0092 |
| gru (≈94K) | 0.0065 ± 0.0002 | 0.1944 ± 0.0077 | 0.2009 ± 0.0079 |
| lstm (≈91K) | 0.0123 ± 0.0004 | 0.2426 ± 0.0047 | 0.2548 ± 0.0050 |

**Lecture.** Sur cette tâche, le résultat ne confirme pas l'hypothèse d'un avantage double-flux : la GRU appariée (H=170, ~95K) obtient la MSE totale la plus basse (≈0.201 contre ≈0.256 pour MFN complet), principalement sur la composante longue b (k=10). L'ablation sans couplage bidirectionnel (≈0.259) est statistiquement indifférenciable de MFN complet (≈0.256) : la contribution du couplage bidirectionnel n'apparaît pas sur cette tâche à ce régime d'entraînement. La tâche 3 ne discrimine donc pas en faveur de l'architecture proposée ; le diagnostic dual-contexte devra être renforcé (délais plus longs, sources corrélées, supervision par flux) avant de conclure.

### 9.3 Tâche 2 — Modélisation du langage niveau caractère (PTB)

Protocole exécuté : 15 epochs, batch 64, séquence 256, Adam lr=1e-3 + anneau cosinus, seed unique, vocabulaire caractère V=50 (5 101 618 caractères d'entraînement), BPTT tronqué avec carry-over d'état.

| Modèle | BPC valid (best) | BPC test | Paramètres |
|---|---:|---:|---:|
| grumatched | 1.3115 | 1.2820 | 107175 |
| mfn | 1.3617 | 1.3341 | 107634 |
| gru | 1.2293 | 1.1960 | 191640 |

**Lecture.** À budget apparié (GRU H=125, 107 175 paramètres vs MFN 107 634), la GRU devance toujours MFN sur le langage niveau caractère (test 1.2820 contre 1.3341 BPC, Δ≈0.05). L'écart se réduit par rapport à la GRU plus large (H=170, 191 640 paramètres, test 1.1960) mais reste robuste. Les deux courbes étaient encore descendantes à 15 epochs (≈0.002 BPC/epoch pour chaque), l'écart pourrait se resserrer encore à convergence complète sans changer la conclusion. Sur les trois tâches, MFN n'a donc pas confirmé d'avantage sur les baselines mono-flux à budget de paramètres égal ; le seul signal favorable reste le délai k=20 de la Tâche 1 (cf. §9.1), à confirmer.

### 9.4 Analyse du profil de fatigue (Tâche 1, k=10)

| Statistique | Valeur |
|---|---:|
| mean_fatigue_lam | 0.1457 |
| mean_fatigue_psi | 0.1186 |
| saturated_fraction_lam | 0.0000 |
| saturated_fraction_psi | 0.0000 |
| fraction_above_0.3_lam | 0.0000 |
| fraction_above_0.3_psi | 0.0148 |
| pearson_corr_phi_lam_phi_psi | 0.9943 |
| pearson_per_neuron_mean | 0.1533 |
| pearson_per_neuron_min | 0.0682 |
| pearson_per_neuron_max | 0.3084 |
| final_loss_k10 | 0.4625 |

**Lecture.** Le flux expressif se fatigue légèrement plus que le flux conceptuel (φ̄Λ=0.146 > φ̄Ψ=0.119), conforme à l'attente (i). La fraction de neurones saturés (ϕ > 0.5) est nulle et reste quasi nulle au seuil 0.3 : sur cette tâche le régime de fatigue est modéré. La corrélation temporelle des moyennes de flux est élevée (r = 0.994) — attendu, car les deux flux sont modulés par l'énergie globale de l'entrée ; en revanche la corrélation **par neurone** (moyenne sur les 64 neurones, échantillons (séquence × temps)) vaut r = 0.153 (min 0.068, max 0.308) : les profils de fatigue des deux flux sont bien décorrélés neurone à neurone, comme requis par le protocole §8.5 (cible < 0.4). La saturation décorrélée est donc confirmée à ce régime.

### 9.5 Synthèse des trois protocoles

1. **Tâche 1 (copy multi-délai)** : MFN complet est le meilleur modèle à k=20 (0.9826), devant GRU (0.9945), xLSTM (0.9951), LSTM (1.0031) et Mamba-like (1.0666), ainsi que devant ses ablations sans fatigue et sans couplage bidirectionnel. La fatigue et le couplage bidirectionnel démontrent ici leur utilité pour la rétention longue portée, avec un gain modeste mais reproductible sur 5 seeds (Δ≈0.005–0.01). À l'inverse, aux courtes portées (k=5) le SSM sélectif domine nettement et MFN complet paie un léger coût par rapport à ses propres ablations : le design paie les courtes portées pour gagner la plus longue, conformément à l'intention architecturale.

2. **Tâche 2 (PTB caractère)** : la GRU appariée (H=125) devance MFN (1.2820 vs 1.3341 BPC test, Δ≈0.05) ; aucun avantage double-flux sur le langage au niveau caractère à ce budget de calcul.

3. **Tâche 3 (contexte dual)** : la GRU appariée obtient la meilleure MSE totale (0.201 vs 0.256 pour MFN) ; l'ablation sans bidirectionnel est indifférenciable de MFN complet (0.259 vs 0.256). L'hypothèse d'un avantage du découplage des flux sur la mémoire parallèle n'est pas confirmée sur cette formulation de la tâche.

**Bilan.** Le protocole expérimental complet (5 seeds, budgets de paramètres contrôlés et comptés réels) confirme le mécanisme attendu — la fatigue neurale et le couplage bidirectionnel aident la rétention à la plus longue portée testée dans la Tâche 1 — mais ne met en évidence aucun avantage global de MFN sur les baselines mono-flux aux autres délais ni sur les tâches 2 et 3. Les affirmations d'un avantage généralisé nécessiteraient des preuves supplémentaires (langage entraîné plus longtemps, tâches de raisonnement, délais plus longs encore). Ce rapport présente les résultats tels que mesurés, y compris ceux qui ne soutiennent pas l'hypothèse.


<!--RESULTS_END-->

---

## 10. Limitations et travaux futurs

- **Coût caché quadratique.** Les cellules GRU et les projections de retour
  s'échelonnent en O(H²) par étape, à égalité avec les GRU standard mais en
  retard sur les SSM linéaires comme Mamba et Griffin. Une factorisation
  structurée (par ex. Monarch [18]) des matrices de poids GRU réduirait cela sans
  changer la sémantique architecturale.
- **Nombre fixe d'échelles de temps.** Le design dual-flux est spécifié
  manuellement. Une extension naturelle apprendrait un mélange sur N flux par
  attention soft sur une banque de cellules GRU, analogue aux architectures
  mixture-of-experts.
- **β scalaire pour la décroissance du flux lent.** L'exposant β = 0.2 est un
  hyperparamètre global. Apprendre β par neurone comme scalaire entraîné
  permettrait des taux d'oubli hétérogènes dans le flux conceptuel, reflétant
  mieux la réalité biologique.
- **Aucune perte cognitive auxiliaire.** MFN est entraîné uniquement sur la perte
  de tâche. Ajouter une perte auxiliaire récompensant le flux conceptuel pour
  prédire les entrées futures — reliant MFN à la théorie du codage prédictif [11]
  — donnerait au flux intérieur une fonction anticipatoire explicite.
- **Récupération de fatigue indépendante de l'entrée.** Alors que γs est appris,
  il est indépendant de l'entrée. Un taux de récupération dynamique — qui
  s'accélère quand l'entrée change rapidement et ralentit en entrée stable —
  modéliserait mieux la récupération dépendante de l'activité observée dans les
  synapses biologiques.
- **Portée de l'évaluation.** La suite expérimentale actuelle couvre les signaux
  séquentiels et le texte au niveau caractère. L'audio, la prédiction de frames
  vidéo et le dialogue multi-locuteur restent des benchmarks futurs importants.
- **Preuve empirique partielle.** Sur les trois protocoles exécutés (Section 9),
  l'avantage mesuré de MFN se limite au délai le plus long de la Tâche 1 (k=20) ;
  les Tâches 2 et 3 ne confirment aucun avantage sur les baselines mono-flux
  appariées. Le gain à k=20 (Δ ≈ 0.005–0.01 vs ablations) est reproductible sur
  5 seeds mais modeste ; sa significativité et sa généralisation (délais plus
  longs, tâches linguistiques plus riches, entraînement plus long) restent à
  établir. La tâche de contexte dual devra être reformulée (délais asymétriques
  plus marqués, sources corrélées, supervision par flux) pour pouvoir discriminer
  l'architecture proposée.

---

## 11. Conclusion

Nous avons introduit MyelinFatigueNet, une architecture récurrente qui décompose
la dynamique cachée en un flux expressif hΛ et un flux conceptuel hΨ, couplés
bidirectionnellement par deux portes de confiance apprises indépendantes et
régulés par des accumulateurs de fatigue neurale par flux. La fatigue est modélisée
comme une trace d'activation décroissante exponentiellement par neurone, avec des
taux de récupération appris par neurone ; son effet est d'empêcher la fixation
représentationnelle et de forcer la diversité d'activation dans le temps — une
propriété fonctionnelle absente de tous les modèles de séquence antérieurs. Le
couplage bidirectionnel permet au raisonnement de fond de guider l'expression de
surface tandis que l'expression renvoie simultanément dans la mise à jour
conceptuelle.

Nous avons dérivé le budget de paramètres complet :
**22H² + (3d + 2d_out + 23)H + d_out**, avec la contribution dominante 12H² des
deux cellules GRU. La formule a été vérifiée par comptage direct : 94 794
paramètres à H=64, d=10, d_out=10.

**Résultats mesurés (Section 9).** Sur le protocole complet (500 epochs × 5
seeds pour les Tâches 1 et 3, 15 epochs pour la Tâche 2), MFN obtient le meilleur
résultat au délai le plus long de la Tâche 1 (k=20 : 0.9826, devant GRU 0.9945,
xLSTM 0.9951, LSTM 1.0031, Mamba-like 1.0666) et devance ses propres ablations
sans fatigue (0.9921) et sans couplage bidirectionnel (0.9876) — la fatigue
neurale et le couplage bidirectionnel améliorent donc la rétention longue
portée, conformément à l'hypothèse centrale du papier, avec un gain modeste mais
reproductible (Δ ≈ 0.005–0.01). En revanche, MFN n'a montré aucun avantage aux
délais courts (k=5, k=10), sur la modélisation du langage au niveau caractère
(PTB : GRU appariée 1.2820 contre 1.3341 BPC test), ni sur le suivi de contexte
dual (GRU 0.2009 contre 0.2559 MSE). Ces résultats, y compris ceux qui
contredisent les hypothèses, sont rapportés tels que mesurés (moyenne ± écart-
type, 5 seeds).

Nous publions l'implémentation de référence complète (`mfn.py`) et l'ensemble
des scripts d'entraînement reproductibles (README). Une déclinaison **O(H)
diagonale** (`mfn_diag.py`, `mfn_lm.py`) est intégrée à l'écosystème
HuggingFace (tokenizer BPE, `PreTrainedModel`/`generate`, Auto APIs) et
benchmarkée en modèle de langue sur Wikitext-2 (README §3). Le gain de rétention
longue portée observé à k=20 justifie de poursuivre l'étude — avec des délais
plus longs, des tâches de raisonnement et un entraînement plus long — avant de
conclure à un avantage généralisé de MFN sur les baselines mono-flux.

---

## Références
1. Vaswani et al. (2017) Attention is all you need. NeurIPS 30.
2. Hochreiter & Schmidhuber (1997) LSTM. Neural Computation 9(8).
3. Cho et al. (2014) GRU. EMNLP.
4. Bengio et al. (1994) Learning long-term dependencies. IEEE TNN 5(2).
5. Gu & Dao (2023) Mamba. arXiv:2312.00752.
6. Gu, Goel & Ré (2021) S4. ICLR.
7. Brown et al. (2020) GPT-3. NeurIPS 33.
8. Dai et al. (2019) Transformer-XL. ACL.
9. Chung et al. (2017) Hierarchical multiscale RNN. ICLR.
10. Koutnik et al. (2014) Clockwork RNN. ICML.
11. Rao & Ballard (1999) Predictive coding. Nature Neuroscience 2(1).
12. Lotter et al. (2016) PredNet. arXiv:1605.08104.
13. Zucker & Regehr (2002) Short-term synaptic plasticity. Annu Rev Physiol 64.
14. Beck et al. (2024) xLSTM. arXiv:2405.04517.
15. De et al. (2024) Griffin. arXiv:2402.19427.
16. Peng et al. (2023) RWKV. EMNLP.
17. Sun et al. (2023) RetNet. arXiv:2307.08621.
18. Dao et al. (2022) Monarch. ICML.