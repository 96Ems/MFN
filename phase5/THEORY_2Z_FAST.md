# Théorie : MFN-2Z-FAST — garder les zones asymétriques, multiplier le débit

Auteur : projet MFN · 23 août 2026 · vérifié par `theory_2z_fast.py`

## 0. Diagnostic chiffré (2Z actuel)

- Coût réel : **1 604 864 MACs/token** (A=1.12M, B=0.39M, latéral=90k)
- **Les 2 GRUCell représentent 45 % du calcul** (721 920 MACs)
- Débit mesuré : 849 tok/s @B32 — soit **~1.1 GFLOP/s efficace** pour un pic GPU
  ~1-2 TFLOP/s → **~0.1 % d'efficacité** : le goulot est le LANCEMENT de ~180
  kernels minuscules par step (3 couches × 2 zones × ~30 ops), pas les FLOPs.

**Théorème (inefficacité séquentielle).** Pour un step de profondeur L, K zones,
c_k ops par couche : le temps wall-clock ∝ L·K·(Σc_k + overhead_launch(GPU)).
Sur cette carte, overhead_launch ≈ 30-50× le temps noyau pour H ≤ 160.

## 1. Astuce 1 — Récurrence-α (fusion GRU × décroissance) : ×1.76

**Observation.** Le GRU a un gate `z` (input-conditionné) et la décroissance
`α = σ(MLP(u))` (input-conditionnée) jouent le MÊME rôle : contrôler la
mutation de l'état. Le papier d'origine (`MyelinFatigueNet`) utilise déjà
`h ← α⊙h + (1−α)⊙tanh(mix)` — la variante dense a ensuite ajouté des GRU qui
dupliquent ce calcul.

**Remplacement (équivalent en capacité) :**
```
h_λ ← α_λ ⊙ h_λ + (1 − α_λ) ⊙ tanh(W_c[ x_p + fb + W_ψ→λ(h̃_ψ) + td + sk ])
```
- `α` sert de gate d'oubli (déjà calculé, déjà conditionné par l'entrée) ;
- la **fatigue** φ fournit la branche « reset » adaptative : `h̃ = h⊙(1−φ)`
  remplace le reset gate `r` (mécanisme de suppression déjà validé).
**Complexité : 12H² → 3H² par stream. Total : 15H² → ~9H² (×1.76, mesuré).**

## 2. Astuce 2 — Sous-échantillonnage temporel de la zone lente : ×1.13 supl.

**Théorème (erreur bornée du hold).** Soit B la zone à β élevé (lente), son
état obéit à `h_B(t+1) = α_B·h_B(t) + (1−α_B)·tanh(...)`. À rate 1/2, on calcule
B à t pair et on **maintient** l'état à t impair. Erreur par saut :
```
‖h_B(t+1) − h_B(t)‖ ≤ (1 − ᾱ_B)·(‖tanh‖ + ‖Δ mix‖) ≤ 2(1−ᾱ_B)
```
Avec ᾱ_B ≈ 0.85-0.95 (β=0.6 → α^0.6 proche de 1), l'erreur est
≤ 0.1-0.3 — et l'erreur est *elle-même* amortie par la dynamique lente
(le ψ-stream est un intégrateur : erreurs décorrélées en temps s'annulent).
**C'est le principe Clockwork-RNN appliqué à une ZONE, pas à des neurones
fixés a priori** : la lenteur est apprise (β par zone).

## 3. Astuce 3 — Latéral bas-rang : ×1.08

`W_lat ∈ R^{h_dst×h_src}` → `W = U·V`, `U∈R^{h_dst×r}`, `V∈R^{r×h_src}`,
`r = min(h_src,h_dst)/2` : coût `h_dst·h_src → r(h_dst+h_src)` (÷3 ici).
Les gates per-neuron `u,v,b` (3H) restent denses — c'est eux qui portent la
spécialisation, pas la projection.

## 4. Astuce 4 — Fusion des kernels (le vrai jackpot)

Sur ce GPU launch-bound : regrouper les opérations par **"paquets" de 2** :
- une seule `Linear(2H, 2H)` pour les 2 portes de confiance ;
- `[fb_λ; fb_ψ]` traités comme batch dans une `Linear(H, 2H)` ;
- `[W_ψ→λ; W_λ→ψ]` idem ;
- décay MLP partageant sa première couche `Linear(h_in, 2H)`.

**Pas de changement mathématique** — les flops identiques, MAIS le nombre de
lancements passe de ~30 à ~12 par couche (×2.5-3 sur le temps).
+ `torch.compile(mode="reduce-overhead")` déjà présent dans train_deep
  (fallback eager non vérifié sur Maxwell → à tester).

## 5. Budget cumulé et réinvestissement de capacité

| Étape | MACs/token | Speedup |
|---|---:|---:|
| 2Z actuel | 1 604 864 | ×1 |
| +α-rec | 913 024 | ×1.76 |
| +B rate 1/2 | 806 208 | ×1.99 |
| +lat bas-rang | 746 133 | ×2.15 |
| +fusion kernels | idem | **×3-6 attendu** |

**849 → 1 826-5 000 tok/s.** Le budget libéré (×2-3) doit être **réinvesti
dans la zone A** (la zone réactive porte la charge token-level) : largeurs
[96,80,64] → [192,128,96] pour A, pendant que B reste étroite-lente → la
**spécialisation s'accentue** (plus d'asymétrie = plus de division du travail)
POUR UN MÊME TEMPS DE MUR. C'est l'axe « capacité/coût » du papier.

## 6. Prédictions falsifiables (à tester après implémentation)

1. **Débit** : 2Z-FAST ≥ 2 500 tok/s @B32 (vs 849), iso-params iso-loss-à-steps-égaux.
2. **Équivalence** : aux steps égaux, 2Z-FAST ≤ 2Z actuel (±0.05 nat) sur les 4
   premiers epochs — les GRU ne portent pas la capacité du 2Z.
3. **Spécialisation préservée** : le probing d'ablation (readout/classe de
   tokens) montre une spécialisation AU MOINS aussi marquée qu'en 2Z dense
   (Δperte par classe ≥).
4. **Erreur du hold** : val(rate2) − val(rate1) ≤ 0.03 nat sur 22.6M tokens.
5. **Complexité effective** : scale-up H_A de [160,112,80] → [192,128,96] à
   iso-temps-de-mur donne une val finale ≤ 2Z dense actuel.

## 7. Implémentation visée (ordre)

1. `MFNZoneLayer` : récurrence-α + fusion (1 ficher, cell unit-tested CPU)
2. `rate` param zone (temporisation de B) dans `MFNDeepStack._step_N`
3. `LateralFBX` bas-rang (flag `lowrank`)
4. Bench `theory_2z_fast.py` étendu (mesures réelles vs prédiction)
5. Run comparatif 2Z vs 2Z-FAST vs 3L @22.6M puis @86M