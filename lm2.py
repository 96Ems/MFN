import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    GenerationMixin,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutput

from mfn_lm import GRUForCausalLM, GRULMConfig, RecurrentStateCacheMixin


# ---------------------------------------------------------------- configs ----

class MFNDenseConfig(PretrainedConfig):
    """Dense dual-stream MFN LM (v2, quality-first): tied head via 2H->H
    bottleneck, LayerNorm, optional dropout. Ablations no_bidir / no_fatigue."""

    model_type = "mfn_dense_lm"

    def __init__(
        self,
        vocab_size: int = 4096,
        hidden_size: int = 144,
        beta: float = 0.2,
        no_bidir: bool = False,
        no_fatigue: bool = False,
        dropout: float = 0.0,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.beta = beta
        self.no_bidir = no_bidir
        self.no_fatigue = no_fatigue
        self.dropout = dropout
        self.initializer_range = initializer_range
        self.tie_word_embeddings = True
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.bos_token_id = 0
        super().__init__(**kwargs)


# ------------------------------------------------------------- dense core ----

class MFNDenseCore(nn.Module):
    """Dense O(H^2) dual-stream core (paper's MyelinFatigueNet equations) with
    honest ablation switches: no_bidir removes the cross-back projections and
    both confidence gates; no_fatigue removes the fatigue accumulators.
    The 2H->H readout is the LM bottleneck (head gets tied to the embedding)."""

    def __init__(self, config):
        super().__init__()
        H = config.hidden_size
        d = H  # LM mode: token embedding is the input representation
        self.H = H
        self.beta = config.beta
        self.no_bidir = config.no_bidir
        self.no_fatigue = config.no_fatigue

        self.input_proj = nn.Linear(d, H)

        self.decay_lam = nn.Sequential(
            nn.Linear(d, H), nn.ReLU(), nn.Linear(H, H), nn.Sigmoid())
        self.decay_psi = nn.Sequential(
            nn.Linear(d, H), nn.ReLU(), nn.Linear(H, H), nn.Sigmoid())

        self.fb_lam = nn.Linear(H, H)
        self.fb_psi = nn.Linear(H, H)

        self.W_psi2lam = nn.Linear(H, H, bias=False)
        if not self.no_bidir:
            self.W_lam2psi = nn.Linear(H, H, bias=False)

        if not self.no_bidir:
            self.gate_psi2lam = nn.Sequential(
                nn.Linear(2 * H, H), nn.Sigmoid())
            self.gate_lam2psi = nn.Sequential(
                nn.Linear(2 * H, H), nn.Sigmoid())

        self.gru_lam = nn.GRUCell(H, H)
        self.gru_psi = nn.GRUCell(H, H)

        if not self.no_fatigue:
            self.w_gamma_lam = nn.Parameter(torch.zeros(H))
            self.w_gamma_psi = nn.Parameter(torch.zeros(H))

        self.readout = nn.Linear(2 * H, H)

    @staticmethod
    def init_state(batch_size, hidden_size, device="cpu"):
        z = lambda: torch.zeros(batch_size, hidden_size, device=device)
        return z(), z(), z(), z()

    def forward(self, x, h_lam, h_psi, phi_lam, phi_psi):
        if self.no_fatigue:
            gamma_lam = gamma_psi = None
            h_lam_eff = h_lam.clone()
            h_psi_eff = h_psi.clone()
        else:
            gamma_lam = torch.sigmoid(self.w_gamma_lam)
            gamma_psi = torch.sigmoid(self.w_gamma_psi)
            h_lam_eff = h_lam * (1.0 - phi_lam).clamp_min(0)
            h_psi_eff = h_psi * (1.0 - phi_psi).clamp_min(0)

        xp = self.input_proj(x)
        # NaN-safe clamp (Proof A1)
        alpha_lam = self.decay_lam(x)
        alpha_psi = self.decay_psi(x).clamp_min(1e-6) ** self.beta

        if self.no_bidir:
            g_p2l = torch.ones_like(h_lam)
            g_l2p = torch.ones_like(h_lam)
        else:
            g_p2l = self.gate_psi2lam(torch.cat([h_psi_eff, h_lam_eff], dim=-1))
            g_l2p = self.gate_lam2psi(torch.cat([h_lam_eff, h_psi_eff], dim=-1))

        fb_lam = self.fb_lam(h_lam_eff) * alpha_lam * g_p2l
        fb_psi = self.fb_psi(h_psi_eff) * alpha_psi * g_l2p

        h_lam = self.gru_lam(
            xp + fb_lam + self.W_psi2lam(h_psi_eff), h_lam)

        h_lam_eff_star = h_lam * (1.0 - phi_lam).clamp_min(0)
        if self.no_bidir:
            wp_to_lam = torch.zeros_like(h_lam)
        else:
            wp_to_lam = self.W_lam2psi(h_lam_eff_star)
        h_psi = self.gru_psi(xp + fb_psi + wp_to_lam, h_psi)

        if not self.no_fatigue:
            phi_lam = (gamma_lam * phi_lam + (1.0 - gamma_lam) * h_lam.abs()).clamp(0, 1)
            phi_psi = (gamma_psi * phi_psi + (1.0 - gamma_psi) * h_psi.abs()).clamp(0, 1)
            h_lam_out = h_lam * (1.0 - phi_lam).clamp_min(0)
            h_psi_out = h_psi * (1.0 - phi_psi).clamp_min(0)
        else:
            h_lam_out = h_lam
            h_psi_out = h_psi
        y = self.readout(torch.cat([h_lam_out, h_psi_out], dim=-1))
        return h_lam, h_psi, phi_lam, phi_psi, y


# ------------------------------------------------------------- LM model -------

class MFNDenseForCausalLM(RecurrentStateCacheMixin, PreTrainedModel, GenerationMixin):
    config_class = MFNDenseConfig
    _tied_weights_keys = {"lm_head.weight": "embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.core = MFNDenseCore(config)
        self.ln = nn.LayerNorm(config.hidden_size)
        self.drop = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
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
        e = self.embed_tokens(input_ids)

        if state is not None:
            # generation step with carried-over recurrent state: one timestep
            h_lam, h_psi, phi_lam, phi_psi = state
            if e.shape[1] > 1:
                e = e[:, -1:]
            h_lam, h_psi, phi_lam, phi_psi, hout = self.core(
                e[:, 0], h_lam, h_psi, phi_lam, phi_psi)
            logits = self.lm_head(self.ln(self.drop(hout)))[:, None]
            return CausalLMOutput(logits=logits,
                                  hidden_states=(h_lam, h_psi, phi_lam, phi_psi))

        h_lam, h_psi, phi_lam, phi_psi = self.core.init_state(
            e.shape[0], self.config.hidden_size, e.device)
        logits = []
        for t in range(e.shape[1]):
            h_lam, h_psi, phi_lam, phi_psi, hout = self.core(
                e[:, t], h_lam, h_psi, phi_lam, phi_psi)
            logits.append(self.lm_head(self.ln(self.drop(hout))))
        logits = torch.stack(logits, dim=1)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

        if not return_dict:
            return ((loss,) if loss is not None else tuple()) + (logits,)
        return CausalLMOutput(
            loss=loss, logits=logits,
            hidden_states=(h_lam, h_psi, phi_lam, phi_psi))

    def prepare_inputs_for_generation(self, input_ids, state=None, **kwargs):
        return {"input_ids": input_ids}

    def _supports_default_dynamic_cache(self) -> bool:
        return False


def build_gpt2mini(vocab_size: int, n_embd: int = 120, n_layer: int = 4,
                   n_head: int = 6):
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=512,
        n_ctx=512,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=4 * n_embd,
        tie_word_embeddings=True,
        eos_token_id=0,
        pad_token_id=0,
        bos_token_id=0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    return GPT2LMHeadModel(cfg)


# ---------------------------------------------------------------- plumbing ----

def true_param_count(model):
    seen = set()
    total = 0
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def build_phase3(model_id: str, vocab_size: int, hidden_size: int,
                 beta: float = 0.2, dropout: float = 0.0):
    if model_id == "mfn_dense":
        cfg = MFNDenseConfig(vocab_size=vocab_size, hidden_size=hidden_size,
                             beta=beta, no_bidir=False, no_fatigue=False,
                             dropout=dropout)
        return MFNDenseForCausalLM(cfg)
    if model_id == "mfn_dense_nobidir":
        cfg = MFNDenseConfig(vocab_size=vocab_size, hidden_size=hidden_size,
                             beta=beta, no_bidir=True, no_fatigue=False,
                             dropout=dropout)
        return MFNDenseForCausalLM(cfg)
    if model_id == "mfn_dense_nofatigue":
        cfg = MFNDenseConfig(vocab_size=vocab_size, hidden_size=hidden_size,
                             beta=beta, no_bidir=False, no_fatigue=True,
                             dropout=dropout)
        return MFNDenseForCausalLM(cfg)
    if model_id == "gpt2mini":
        return build_gpt2mini(vocab_size, n_embd=hidden_size)
    if model_id == "gru":
        cfg = GRULMConfig(vocab_size=vocab_size, hidden_size=hidden_size,
                          layernorm=True)
        return GRUForCausalLM(cfg)
    raise ValueError(f"unknown model id: {model_id}")


def matched_gru_hidden(target_params: int, vocab_size: int) -> int:
    for H in range(1, 2048):
        p = vocab_size * H + 3 * (2 * H * H + 2 * H) + 2 * H  # + LN scales
        if p >= target_params:
            return H
    raise ValueError("no H found")


def register_auto():
    AutoConfig.register(MFNDenseConfig.model_type, MFNDenseConfig)
    AutoModelForCausalLM.register(MFNDenseConfig, MFNDenseForCausalLM)


register_auto()
MFNDenseConfig.register_for_auto_class("AutoConfig")
MFNDenseForCausalLM.register_for_auto_class("AutoModelForCausalLM")