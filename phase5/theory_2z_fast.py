"""Théorie : coût/step du MFN-2Z et gains attendus des 4 astuces (math pure)."""
# Complexité par couche/zone (MACs par stream paire) :
#   GRUcell: 12H^2 ; gates conf: 4H^2 ; fb: 2H^2 ; cross: 2H^2 ;
#   decay MLPs: ~2H^2 (+entrées) ; readout: 2H^2 ; input_proj: h_in*H
H_A = [160, 112, 80]; H_B = [96, 64, 48]   # zones A (réactive), B (intégrative)
def layer_cost(H, H_in):
    return (12*H*H        # GRUs (dominant)
            + 4*H*H       # portes de confiance 2H->H x2
            + 2*H*H + 2*H*H  # fb + cross
            + 2*H*H       # decay MLPs (approx)
            + 2*H*H       # readout
            + H*H_in)     # input proj
def zone_cost(Hs):
    Hs_in = [Hs[0]] + Hs[:-1]
    return sum(layer_cost(h, hi) for h, hi in zip(Hs, Hs_in))
CA, CB = zone_cost(H_A), zone_cost(H_B)
lat = 2 * sum(H_A[l]*H_B[l] for l in range(2)) * 2  # 2 étages x 2 sens
tot = CA + CB + lat
print(f"MACs/token  A={CA:,}  B={CB:,}  lat={lat:,}  total={tot:,}")
print(f"GRU part: {sum(12*h*h for h in H_A+H_B):,} = {sum(12*h*h for h in H_A+H_B)/tot*100:.0f}% du total")
# Astuce 1 : recurrence alpha (dépose GRU) : 12H^2 -> 3H^2 (cand + 1 gate partagé)
def layer_cost_alpha(H, H_in):
    return (3*H*H        # candidate tanh W([x;h]) + mutation alpha reuse
            + 2*H*H      # fb
            + 2*H*H      # cross
            + 2*H*H      # gates (fusion 1 Linear(2H,2H))
            + 1.5*H*H    # decay partagé
            + 2*H*H      # readout
            + H*H_in)
def zone_cost_a(Hs):
    Hs_in = [Hs[0]] + Hs[:-1]
    return sum(layer_cost_alpha(h, hi) for h, hi in zip(Hs, Hs_in))
tot_a = zone_cost_a(H_A) + zone_cost_a(H_B) + lat
print(f"Apres alpha-rec: total={tot_a:,}  -> speedup MACs = {tot/tot_a:.2f}x")
# Astuce 2 : sous-echantillonnage temporel de la zone B (hold 1 step / 2)
tot_as = zone_cost_a(H_A) + zone_cost_a(H_B)/2 + lat
print(f"+ B sur 1 pas/2: total={tot_as:,} -> speedup = {tot/tot_as:.2f}x")
# Astuce 3 : lateral bas-rang r=H_min/2
lat_lr = lat/3
tot_asl = zone_cost_a(H_A) + zone_cost_a(H_B)/2 + lat_lr
print(f"+ lat bas-rang: total={tot_asl:,} -> speedup cumule {tot/tot_asl:.2f}x")
print("\nPrediction: 2Z actuel 849 tok/s -> ~{:.0f}-{:.0f} tok/s (fusion kernels exclue)".format(
    849*tot/tot_asl, 849*tot/tot_asl*1.5))
