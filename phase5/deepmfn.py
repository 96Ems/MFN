"""MFN deep stack with inter-layer feedback (phase 5).

Each layer is a dual-stream core (lambda/fast + psi/slow, per the paper) of
configurable width H_l. Two extra couplings, in the spirit of the paper's
bidirectional confidence gating but *between layers*:

  - top-down:  the deeper layer's readout y_{l+1} (at t-1) is projected to
               H_l and added (per-neuron gated) to both streams of layer l;
  - skip:      with skip_fb=True, layer 0's readout also feeds non-adjacent
               deeper layers (surface -> deep long-range feedback).

The layer-0 width fixes the embedding/head dims; the final layer's 2H_L
readout is bottlenecked to H_0 so the LM head stays tied to the embedding.
"""
import torch
import torch.nn as nn
from transformers import (
    AutoConfig, AutoModelForCausalLM, GenerationMixin, PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutput

from mfn_lm import RecurrentStateCacheMixin


class MFNDeepConfig(PretrainedConfig):
    model_type = "mfn_deep_lm"

    def __init__(
        self,
        vocab_size: int = 4096,
        layer_widths=None,
        beta: float = 0.2,
        topdown: bool = True,
        skip_fb: bool = False,
        n_threads: int = 1,
        thread_fb: bool = True,
        dropout: float = 0.0,
        initializer_range: float = 0.02,
        zone_widths=None,
        zone_betas=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.layer_widths = list(layer_widths) if layer_widths else [192, 96]
        self.beta = beta
        # --- zones asymétriques (MFN-2Z) : thread t = zone t ---
        # zone_widths[t] = largeurs de la zone t (même nb d'étages L requis);
        # zone_betas[t] = beta du psi-stream de la zone t (vitesse de conduction).
        self.zone_widths = [list(z) for z in zone_widths] if zone_widths else None
        self.zone_betas = list(zone_betas) if zone_betas else None
        self.topdown = topdown
        self.skip_fb = skip_fb
        self.n_threads = n_threads
        self.thread_fb = thread_fb
        self.dropout = dropout
        self.initializer_range = initializer_range
        self.tie_word_embeddings = True
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.bos_token_id = 0
        super().__init__(**kwargs)


class MFNDeepLayer(nn.Module):
    """One dual-stream layer of width H with optional inter-layer feedback."""

    def __init__(self, h_in, H, h_td=None, h_skip=None, beta=0.2):
        super().__init__()
        self.H = H
        self.beta = beta
        self.input_proj = nn.Linear(h_in, H)

        self.decay_lam = nn.Sequential(
            nn.Linear(h_in, H), nn.ReLU(), nn.Linear(H, H), nn.Sigmoid())
        self.decay_psi = nn.Sequential(
            nn.Linear(h_in, H), nn.ReLU(), nn.Linear(H, H), nn.Sigmoid())

        self.fb_lam = nn.Linear(H, H)
        self.fb_psi = nn.Linear(H, H)
        self.W_psi2lam = nn.Linear(H, H, bias=False)
        self.W_lam2psi = nn.Linear(H, H, bias=False)
        self.gate_psi2lam = nn.Sequential(nn.Linear(2 * H, H), nn.Sigmoid())
        self.gate_lam2psi = nn.Sequential(nn.Linear(2 * H, H), nn.Sigmoid())

        self.gru_lam = nn.GRUCell(H, H)
        self.gru_psi = nn.GRUCell(H, H)
        self.w_gamma_lam = nn.Parameter(torch.zeros(H))
        self.w_gamma_psi = nn.Parameter(torch.zeros(H))

        self.readout = nn.Linear(2 * H, H)

        # inter-layer couplings (per-neuron confidence gates, diagonal)
        if h_td is not None:
            self.W_td = nn.Linear(h_td, H)
            self.td_lam_u = nn.Parameter(torch.zeros(H))
            self.td_lam_v = nn.Parameter(torch.zeros(H))
            self.td_lam_b = nn.Parameter(torch.zeros(H))
            self.td_psi_u = nn.Parameter(torch.zeros(H))
            self.td_psi_v = nn.Parameter(torch.zeros(H))
            self.td_psi_b = nn.Parameter(torch.zeros(H))
        if h_skip is not None:
            self.W_sk = nn.Linear(h_skip, H)
            self.sk_lam_u = nn.Parameter(torch.zeros(H))
            self.sk_lam_v = nn.Parameter(torch.zeros(H))
            self.sk_lam_b = nn.Parameter(torch.zeros(H))
            self.sk_psi_u = nn.Parameter(torch.zeros(H))
            self.sk_psi_v = nn.Parameter(torch.zeros(H))
            self.sk_psi_b = nn.Parameter(torch.zeros(H))

    @staticmethod
    def init_state(batch, H, device="cpu"):
        z = lambda: torch.zeros(batch, H, device=device)
        return z(), z(), z(), z(), z()  # h_lam, h_psi, phi_lam, phi_psi, y

    def forward(self, u, st, y_td, y_sk):
        """u: input (h_in); st: (h_lam, h_psi, phi_lam, phi_psi, y_prev);
        y_td / y_sk: feedback readouts (or None)."""
        h_lam, h_psi, phi_lam, phi_psi, _ = st
        gamma_lam = torch.sigmoid(self.w_gamma_lam)
        gamma_psi = torch.sigmoid(self.w_gamma_psi)
        h_lam_eff = h_lam * (1.0 - phi_lam).clamp_min(0)
        h_psi_eff = h_psi * (1.0 - phi_psi).clamp_min(0)

        xp = self.input_proj(u)
        alpha_lam = self.decay_lam(u)
        # FIX NaN backward : sigmoid(z)**beta a une dérivée infinie quand
        # sigmoid -> 0.0 (fp32), ce qui infecte le backward en NaN (zone de
        # divergence ~step2000, puis ~97% des steps en ep3-4). clamp_min garde
        # le forward identique (x^0.2 ≈ 0 pour x<1e-6) et tue le pic de dérivée.
        alpha_psi = self.decay_psi(u).clamp_min(1e-6) ** self.beta

        g_p2l = self.gate_psi2lam(torch.cat([h_psi_eff, h_lam_eff], dim=-1))
        g_l2p = self.gate_lam2psi(torch.cat([h_lam_eff, h_psi_eff], dim=-1))
        fb_lam = self.fb_lam(h_lam_eff) * alpha_lam * g_p2l
        fb_psi = self.fb_psi(h_psi_eff) * alpha_psi * g_l2p

        mix_lam = xp + fb_lam + self.W_psi2lam(h_psi_eff)
        if y_td is not None:  # top-down: deeper layer (t-1)
            td = self.W_td(y_td)
            g = torch.sigmoid(self.td_lam_u * h_lam_eff
                              + self.td_lam_v * td + self.td_lam_b)
            mix_lam = mix_lam + g * td
        if y_sk is not None:  # skip: layer-0 readout (same step)
            sk = self.W_sk(y_sk)
            g = torch.sigmoid(self.sk_lam_u * h_lam_eff
                              + self.sk_lam_v * sk + self.sk_lam_b)
            mix_lam = mix_lam + g * sk

        h_lam = self.gru_lam(mix_lam, h_lam)

        # psi uses the FRESH lambda state; feedback terms enter psi's mix too
        h_lam_star = h_lam * (1.0 - phi_lam).clamp_min(0)
        mix_psi = xp + fb_psi + self.W_lam2psi(h_lam_star)
        if y_td is not None:
            g = torch.sigmoid(self.td_psi_u * h_psi_eff
                              + self.td_psi_v * td + self.td_psi_b)
            mix_psi = mix_psi + g * td
        if y_sk is not None:
            g = torch.sigmoid(self.sk_psi_u * h_psi_eff
                              + self.sk_psi_v * sk + self.sk_psi_b)
            mix_psi = mix_psi + g * sk

        h_psi = self.gru_psi(mix_psi, h_psi)

        phi_lam = (gamma_lam * phi_lam + (1.0 - gamma_lam) * h_lam.abs()).clamp(0, 1)
        phi_psi = (gamma_psi * phi_psi + (1.0 - gamma_psi) * h_psi.abs()).clamp(0, 1)
        h_lam_out = h_lam * (1.0 - phi_lam).clamp_min(0)
        h_psi_out = h_psi * (1.0 - phi_psi).clamp_min(0)
        y = self.readout(torch.cat([h_lam_out, h_psi_out], dim=-1))
        return (h_lam, h_psi, phi_lam, phi_psi, y)


class LateralFB(nn.Module):
    """Feedback latéral entre threads (même recette que le topdown intra-thread :
    projection + gate per-neuron sigmoïde sur (état_self, état_other))."""

    def __init__(self, H):
        super().__init__()
        self.W = nn.Linear(H, H)
        self.u = nn.Parameter(torch.zeros(H))
        self.v = nn.Parameter(torch.zeros(H))
        self.b = nn.Parameter(torch.zeros(H))

    def forward(self, h_me, h_other):
        g = torch.sigmoid(self.u * h_me + self.v * h_other + self.b)
        return g * self.W(h_other)


class LateralFBX(nn.Module):
    """Feedback latéral entre ZONES de largeurs différentes (MFN-2Z).
    Projection d'abord (h_src -> h_dst), puis gate per-neuron sur
    (y_me fixe, projection) — ordre-indépendant (fix A2)."""

    def __init__(self, h_src, h_dst):
        super().__init__()
        self.W = nn.Linear(h_src, h_dst)
        self.u = nn.Parameter(torch.zeros(h_dst))
        self.v = nn.Parameter(torch.zeros(h_dst))
        self.b = nn.Parameter(torch.zeros(h_dst))

    def forward(self, y_me, y_other):
        p = self.W(y_other)
        g = torch.sigmoid(self.u * y_me + self.v * p + self.b)
        return g * p


class MFNDeepStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        zw = getattr(config, "zone_widths", None)
        self.zone_mode = bool(zw) and getattr(config, "n_threads", 1) > 1
        if self.zone_mode:
            # --- MFN-2Z : chaque thread = une zone (largeurs/beta propres) ---
            zs = [list(z) for z in zw]
            L = len(zs[0])
            assert all(len(z) == L for z in zs), "zones: même nb d'étages requis"
            self.widths = zs[0]              # référence = zone 0 (emb dim)
            self.zone_widths = zs
            self.n_threads = config.n_threads
            self.thread_fb = getattr(config, "thread_fb", True)
            betas = getattr(config, "zone_betas", None) or [config.beta] * self.n_threads
            emb_dim = zs[0][0]
            self.stacks = nn.ModuleList()
            for t in range(self.n_threads):
                w_t, H0t = zs[t], zs[t][0]
                layers = nn.ModuleList()
                for l in range(L):
                    h_in = emb_dim if l == 0 else w_t[l - 1]
                    h_td = w_t[l + 1] if (config.topdown and l < L - 1) else None
                    h_sk = w_t[0] if (config.skip_fb and l >= 2) else None
                    layers.append(MFNDeepLayer(h_in, w_t[l], h_td, h_sk,
                                               beta=betas[t]))
                self.stacks.append(layers)
            # lat[l][dst][src] entre zones de largeurs différentes
            self.lat = nn.ModuleList()
            if self.thread_fb and self.n_threads > 1:
                for l in range(L - 1):
                    pairs = nn.ModuleList()
                    for dst in range(self.n_threads):
                        pairs.append(nn.ModuleList(
                            [LateralFBX(zs[src][l], zs[dst][l])
                             for src in range(self.n_threads)]))
                    self.lat.append(pairs)
            head_in = 2 * sum(z[-1] for z in zs)
            self.final_proj = nn.Linear(head_in, emb_dim)
            return
        w = config.layer_widths
        self.widths = w
        self.n_threads = getattr(config, "n_threads", 1)
        self.thread_fb = getattr(config, "thread_fb", True)
        if self.n_threads == 1:
            # structure EXACTE d'origine (stack.layers) pour que les
            # checkpoints 2l/3l existants se chargent à l'identique
            self.layers = nn.ModuleList()
            for l, H in enumerate(w):
                h_in = w[l - 1] if l > 0 else H
                h_td = w[l + 1] if (config.topdown and l < len(w) - 1) else None
                h_sk = w[0] if (config.skip_fb and l >= 2) else None
                self.layers.append(
                    MFNDeepLayer(h_in, H, h_td, h_sk, config.beta))
        else:
            # grille : stacks[t][l] = couche l du thread t (autonome,
            # topdown+skip intra-thread) + couplage latéral entre threads.
            self.stacks = nn.ModuleList()
            for _ in range(self.n_threads):
                layers = nn.ModuleList()
                for l, H in enumerate(w):
                    h_in = w[l - 1] if l > 0 else H
                    h_td = w[l + 1] if (config.topdown and l < len(w) - 1) else None
                    h_sk = w[0] if (config.skip_fb and l >= 2) else None
                    layers.append(
                        MFNDeepLayer(h_in, H, h_td, h_sk, config.beta))
                self.stacks.append(layers)
        # lat[l][dst][src] : feedback du thread src vers dst à la sortie de la
        # couche l (largeur w[l]).
        self.lat = nn.ModuleList()
        if self.thread_fb and self.n_threads > 1:
            for l in range(len(w) - 1):
                pairs = nn.ModuleList()
                for _ in range(self.n_threads):
                    pairs.append(nn.ModuleList(
                        [LateralFB(w[l]) for _ in range(self.n_threads)]))
                self.lat.append(pairs)
        self.final_proj = nn.Linear(2 * w[-1] * self.n_threads, w[0])

    def init_state(self, batch, device="cpu"):
        if self.n_threads == 1:
            return [lay.init_state(batch, lay.H, device)
                    for lay in self.layers]
        return [[lay.init_state(batch, lay.H, device) for lay in st]
                for st in self.stacks]

    def _step_1(self, emb, states):
        """Step original (n_threads=1), bit-identique à l'avant-grillage."""
        new_states, y_l = [], None
        y0_fresh = None
        for l, (lay, st) in enumerate(zip(self.layers, states)):
            u = emb if l == 0 else y_l
            y_td = states[l + 1][4] if hasattr(lay, "W_td") else None
            y_sk = y0_fresh if hasattr(lay, "W_sk") else None
            st_new = lay(u, st, y_td, y_sk)
            new_states.append(st_new)
            y_l = st_new[4]
            if l == 0:
                y0_fresh = st_new[4]
        return new_states, y_l

    def _step_N(self, emb, states):
        """Step grillage (n_threads>1) : tous les threads calculent la couche
        l, puis feedback latéral mélange leurs y frais (u de la couche l+1)."""
        L = len(self.widths)
        new_threads = []
        y_l = [None] * self.n_threads
        y0_fresh = [None] * self.n_threads
        for l in range(L):
            st_new_t = []
            for t in range(self.n_threads):
                lay = self.stacks[t][l]
                u = emb if l == 0 else y_l[t]
                y_td = states[t][l + 1][4] if hasattr(lay, "W_td") else None
                y_sk = y0_fresh[t] if hasattr(lay, "W_sk") else None
                st_new_t.append(lay(u, states[t][l], y_td, y_sk))
            if l == 0:
                y0_fresh = [st_new_t[t][4] for t in range(self.n_threads)]
            if self.thread_fb and l < L - 1:
                # Fix A2 (proof /tmp/proof_A2_lateral.py): gate must see fixed y_self, not accumulating acc.
                # Previously acc threaded through gate -> order-dependent (Δ0.06). Now y_self fixed -> order-independent, matches Eq. lateral.
                mixed = []
                for t in range(self.n_threads):
                    y_self = st_new_t[t][4]
                    acc = y_self
                    for s in range(self.n_threads):
                        if s != t:
                            acc = acc + self.lat[l][t][s](
                                y_self, st_new_t[s][4])
                    mixed.append(acc)
                y_l = mixed
            else:
                y_l = [st_new_t[t][4] for t in range(self.n_threads)]
            new_threads.append(st_new_t)
        return [[new_threads[l][t] for l in range(L)]
                for t in range(self.n_threads)], y_l

    def step(self, emb, states):
        if self.n_threads == 1:
            return self._step_1(emb, states)
        return self._step_N(emb, states)

    def combine(self, states):
        """Concatène les sorties (2*hidden final) des threads -> final_proj."""
        if self.n_threads == 1:
            last = states[-1]
            return torch.cat([last[0] * (1 - last[2]),
                              last[1] * (1 - last[3])], dim=-1)
        hcats = []
        for t in range(self.n_threads):
            last = states[t][-1]
            hcats.append(last[0] * (1 - last[2]))
            hcats.append(last[1] * (1 - last[3]))
        return torch.cat(hcats, dim=-1)


class MFNDeepForCausalLM(RecurrentStateCacheMixin, PreTrainedModel, GenerationMixin):
    config_class = MFNDeepConfig
    _tied_weights_keys = {"lm_head.weight": "embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        H0 = config.layer_widths[0]
        self.embed_tokens = nn.Embedding(config.vocab_size, H0)
        self.stack = MFNDeepStack(config)
        self.ln = nn.LayerNorm(H0)
        self.lm_head = nn.Linear(H0, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(self, input_ids=None, state=None, labels=None,
                return_dict=None, attention_mask=None, **kwargs):
        return_dict = return_dict if return_dict is not None else True
        emb = self.embed_tokens(input_ids)

        if state is not None:
            if emb.shape[1] > 1:
                emb = emb[:, -1:]
            states, _ = self.stack.step(emb[:, 0], state)
            hout = self.stack.final_proj(self.stack.combine(states))
            logits = self.lm_head(self.ln(hout))[:, None]
            return CausalLMOutput(logits=logits, hidden_states=tuple(states))

        states = self.stack.init_state(emb.shape[0], emb.device)
        logits = []
        for t in range(emb.shape[1]):
            states, _ = self.stack.step(emb[:, t], states)
            hout = self.stack.final_proj(self.stack.combine(states))
            logits.append(self.lm_head(self.ln(hout)))
        logits = torch.stack(logits, dim=1)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        if not return_dict:
            return ((loss,) if loss is not None else tuple()) + (logits,)
        return CausalLMOutput(loss=loss, logits=logits,
                              hidden_states=tuple(states))


def build_deep(model_id: str, vocab_size: int, widths, beta=0.2,
               topdown=True, skip_fb=False, n_threads=1, thread_fb=True,
               zone_widths=None, zone_betas=None):
    cfg = MFNDeepConfig(vocab_size=vocab_size, layer_widths=widths, beta=beta,
                        topdown=topdown, skip_fb=skip_fb,
                        n_threads=n_threads, thread_fb=thread_fb,
                        zone_widths=zone_widths, zone_betas=zone_betas)
    return MFNDeepForCausalLM(cfg)


def register_auto():
    AutoConfig.register(MFNDeepConfig.model_type, MFNDeepConfig)
    AutoModelForCausalLM.register(MFNDeepConfig, MFNDeepForCausalLM)


register_auto()
MFNDeepConfig.register_for_auto_class("AutoConfig")
MFNDeepForCausalLM.register_for_auto_class("AutoModelForCausalLM")