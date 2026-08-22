import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GenerationMixin,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutput


# ---------------------------------------------------------------- configs ----

class MFNDiagConfig(PretrainedConfig):
    """Configuration for the O(H) diagonal-fusion MFN language model."""

    model_type = "mfn_diag_lm"

    def __init__(
        self,
        vocab_size: int = 4096,
        hidden_size: int = 256,
        beta: float = 0.2,
        no_bidir: bool = False,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.beta = beta
        self.no_bidir = no_bidir
        self.initializer_range = initializer_range
        self.tie_word_embeddings = True
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.bos_token_id = 0
        super().__init__(**kwargs)


class GRULMConfig(PretrainedConfig):
    """Configuration for the GRU baseline language model (matched params)."""

    model_type = "gru_lm"

    def __init__(
        self,
        vocab_size: int = 4096,
        hidden_size: int = 199,
        initializer_range: float = 0.02,
        layernorm: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.layernorm = layernorm
        self.tie_word_embeddings = True
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.bos_token_id = 0
        super().__init__(**kwargs)


# ------------------------------------------------------------- diag cell ----

class MyelinFatigueDiagCell(nn.Module):
    """O(H) dual-stream cell adapted for language modelling.

    LM adaptation of the O(H) diagonal variant MyelinFatigueNetDiag: the token
    embedding e_t (dim H) is the input representation, so the dense d->H input
    projections (input_proj and the two decay MLP first layers) reduce to
    elementwise operations (scale/bias per neuron, conditioned on e_t). Every
    other dense HxH interaction is already diagonal in the O(H) variant.
    """

    def __init__(self, config):
        super().__init__()
        H = config.hidden_size
        self.H = H
        self.beta = config.beta
        self.no_bidir = config.no_bidir

        # Decay (input-conditioned, per-neuron): alpha = sigmoid(scale * e_t + bias)
        self.decay_lam_scale = nn.Parameter(torch.ones(H))
        self.decay_lam_bias = nn.Parameter(torch.zeros(H))
        self.decay_psi_scale = nn.Parameter(torch.ones(H))
        self.decay_psi_bias = nn.Parameter(torch.zeros(H))

        # Feedback: diagonal
        self.fb_lam_diag = nn.Parameter(torch.randn(H) * 0.1)
        self.fb_psi_diag = nn.Parameter(torch.randn(H) * 0.1)

        # Cross-stream: diagonal
        self.W_psi2lam_diag = nn.Parameter(torch.randn(H) * 0.1)
        if self.no_bidir:
            self.W_lam2psi_diag = 0.0
        else:
            self.W_lam2psi_diag = nn.Parameter(torch.randn(H) * 0.1)

        # Bidirectional gates: diagonal on each source
        if not self.no_bidir:
            self.gate_p2l_u = nn.Parameter(torch.randn(H) * 0.1)
            self.gate_p2l_v = nn.Parameter(torch.randn(H) * 0.1)
            self.gate_p2l_b = nn.Parameter(torch.zeros(H))
            self.gate_l2p_u = nn.Parameter(torch.randn(H) * 0.1)
            self.gate_l2p_v = nn.Parameter(torch.randn(H) * 0.1)
            self.gate_l2p_b = nn.Parameter(torch.zeros(H))

        self.w_gamma_lam = nn.Parameter(torch.zeros(H))
        self.w_gamma_psi = nn.Parameter(torch.zeros(H))

    @staticmethod
    def init_state(batch_size, hidden_size, device="cpu"):
        z = lambda: torch.zeros(batch_size, hidden_size, device=device)
        return z(), z(), z(), z()

    def forward(self, e, h_lam, h_psi, phi_lam, phi_psi):
        gamma_lam = torch.sigmoid(self.w_gamma_lam)
        gamma_psi = torch.sigmoid(self.w_gamma_psi)

        h_lam_eff = h_lam * (1.0 - phi_lam).clamp_min(0)
        h_psi_eff = h_psi * (1.0 - phi_psi).clamp_min(0)

        # NaN-safe (Proof A1): clamp bounds gradient to 1.3e4
        alpha_lam = torch.sigmoid(self.decay_lam_scale * e + self.decay_lam_bias)
        alpha_psi = torch.sigmoid(self.decay_psi_scale * e + self.decay_psi_bias).clamp_min(1e-6) ** self.beta

        if self.no_bidir:
            g_p2l = torch.ones_like(h_lam)
            g_l2p = torch.ones_like(h_lam)
        else:
            g_p2l = torch.sigmoid(
                self.gate_p2l_u * h_psi_eff + self.gate_p2l_v * h_lam_eff + self.gate_p2l_b
            )
            g_l2p = torch.sigmoid(
                self.gate_l2p_u * h_lam_eff + self.gate_l2p_v * h_psi_eff + self.gate_l2p_b
            )

        fb_lam = self.fb_lam_diag * h_lam_eff * alpha_lam * g_p2l
        fb_psi = self.fb_psi_diag * h_psi_eff * alpha_psi * g_l2p

        mix_lam = e + fb_lam + self.W_psi2lam_diag * h_psi_eff
        h_lam = alpha_lam * h_lam + (1.0 - alpha_lam) * torch.tanh(mix_lam)

        h_lam_eff_star = h_lam * (1.0 - phi_lam).clamp_min(0)
        mix_psi = e + fb_psi + self.W_lam2psi_diag * h_lam_eff_star
        h_psi = alpha_psi * h_psi + (1.0 - alpha_psi) * torch.tanh(mix_psi)

        phi_lam = (gamma_lam * phi_lam + (1.0 - gamma_lam) * h_lam.abs()).clamp(0, 1)
        phi_psi = (gamma_psi * phi_psi + (1.0 - gamma_psi) * h_psi.abs()).clamp(0, 1)

        return h_lam, h_psi, phi_lam, phi_psi


# --------------------------------------------------------- causal LM models ----

class RecurrentStateCacheMixin:
    """State-carry generation for recurrent LMs.

    Instead of recomputing the whole prefix at each generated token (default
    cache-less behaviour, O(n^2) cell steps), the hidden state is threaded
    through model_kwargs and only the last token is processed per step.
    Subclasses must accept `state` in forward() and return it in
    outputs.hidden_states.
    """

    def prepare_inputs_for_generation(self, input_ids, state=None, **kwargs):
        if state is not None:
            return {"input_ids": input_ids[:, -1:], "state": state}
        return {"input_ids": input_ids}

    def _update_model_kwargs_for_generation(
        self, outputs, model_kwargs, is_encoder_decoder=False,
        num_new_tokens=1,
    ):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens)
        if getattr(outputs, "hidden_states", None) is not None:
            model_kwargs["state"] = outputs.hidden_states
        return model_kwargs

    def _supports_default_dynamic_cache(self) -> bool:
        return False

    @torch.no_grad()
    def generate_stream(self, input_ids, max_new_tokens=100, temperature=0.8,
                        top_k=40, repetition_penalty=1.0, eos_token_id=None,
                        greedy=False):
        """Fast O(n) generation with carried-over recurrent state.

        The prompt is processed once (prefill); each generated token then costs
        a single cell step, instead of recomputing the whole prefix at every
        step. Returns the full sequence (prompt + generated).
        """
        cur = input_ids
        out = self(cur, state=None, return_dict=True)
        logits = out.logits[:, -1]
        eos = eos_token_id if eos_token_id is not None else getattr(
            self.config, "eos_token_id", None)
        for _ in range(max_new_tokens):
            if repetition_penalty != 1.0:
                for b in range(logits.shape[0]):
                    for tid in cur[b].unique().tolist():
                        if tid >= logits.shape[1]:
                            continue
                        p = logits[b, tid]
                        logits[b, tid] = p / repetition_penalty if p > 0 \
                            else p * repetition_penalty
            if greedy or temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                l = logits / temperature
                if top_k and top_k > 0 and l.shape[1] > top_k:
                    kth = torch.topk(l, top_k, dim=-1).values[:, -1:]
                    l = l.masked_fill(l < kth, float("-inf"))
                nxt = torch.multinomial(torch.softmax(l, dim=-1), num_samples=1)
            cur = torch.cat([cur, nxt], dim=1)
            if eos is not None and (nxt == eos).all():
                break
            out = self(cur[:, -1:], state=out.hidden_states, return_dict=True)
            logits = out.logits[:, -1]
        return cur


class MFNDiagForCausalLM(RecurrentStateCacheMixin, PreTrainedModel, GenerationMixin):
    """O(H) MyelinFatigueNet with LM head, transformers-compatible.

    Recurrent, recomputes the whole prefix at each generation step (fine for
    short generations). Embedding / LM head weights are tied.
    """

    config_class = MFNDiagConfig
    _tied_weights_keys = {"lm_head.weight": "embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.cell = MyelinFatigueDiagCell(config)
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
        e = self.embed_tokens(input_ids)  # (B, T, H)

        if state is not None:
            h_lam, h_psi, phi_lam, phi_psi = state
            if e.shape[1] > 1:
                e = e[:, -1:]
            h_lam, h_psi, phi_lam, phi_psi = self.cell(
                e[:, 0], h_lam, h_psi, phi_lam, phi_psi)
            out = (h_lam * (1.0 - phi_lam)) + (h_psi * (1.0 - phi_psi))
            logits = self.lm_head(out)[:, None]
            return CausalLMOutput(logits=logits,
                                  hidden_states=(h_lam, h_psi, phi_lam, phi_psi))

        h_lam, h_psi, phi_lam, phi_psi = self.cell.init_state(
            input_ids.shape[0], self.config.hidden_size, input_ids.device)
        h_lam = h_lam.expand(e.shape[0], -1).contiguous()
        h_psi = h_psi.expand(e.shape[0], -1).contiguous()
        phi_lam = phi_lam.expand(e.shape[0], -1).contiguous()
        phi_psi = phi_psi.expand(e.shape[0], -1).contiguous()

        logits = []
        for t in range(e.shape[1]):
            h_lam, h_psi, phi_lam, phi_psi = self.cell(
                e[:, t], h_lam, h_psi, phi_lam, phi_psi)
            out = (h_lam * (1.0 - phi_lam)) + (h_psi * (1.0 - phi_psi))
            logits.append(self.lm_head(out))
        logits = torch.stack(logits, dim=1)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

        if not return_dict:
            return ((loss,) if loss is not None else tuple()) + (logits,)

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=(h_lam, h_psi, phi_lam, phi_psi),
        )


class GRUForCausalLM(RecurrentStateCacheMixin, PreTrainedModel, GenerationMixin):
    """GRU baseline with LM head, transformers-compatible, embeddings tied."""

    config_class = GRULMConfig
    _tied_weights_keys = {"lm_head.weight": "embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        H = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, H)
        self.gru_cell = nn.GRUCell(H, H)
        if config.layernorm:
            self.ln = nn.LayerNorm(H)
        self.lm_head = nn.Linear(H, config.vocab_size, bias=False)
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
            (h,) = state
            if e.shape[1] > 1:
                e = e[:, -1:]
            h = self.gru_cell(e[:, 0], h)
            hh = self.ln(h) if hasattr(self, "ln") else h
            logits = self.lm_head(hh)[:, None]
            return CausalLMOutput(logits=logits, hidden_states=(h,))

        h = torch.zeros(e.shape[0], self.config.hidden_size, device=e.device)
        logits = []
        for t in range(e.shape[1]):
            h = self.gru_cell(e[:, t], h)
            hh = self.ln(h) if hasattr(self, "ln") else h
            logits.append(self.lm_head(hh))
        logits = torch.stack(logits, dim=1)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

        if not return_dict:
            return ((loss,) if loss is not None else tuple()) + (logits,)

        return CausalLMOutput(loss=loss, logits=logits, hidden_states=(h,))


class DenseMFNConfig(MFNDiagConfig):
    """Configuration for the O(H^2) dense MFN reference LM (untied head)."""

    model_type = "mfn_dense_lm"

    def __init__(self, **kwargs):
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)


# ------------------------------------------------------- dense MFN reference ----

class DenseMFNForCausalLM(RecurrentStateCacheMixin, PreTrainedModel, GenerationMixin):
    """O(H^2) dense MFN (original architecture) as LM, for the speed/quality
    reference. Readout is not tied to the embedding (2H -> V)."""

    config_class = DenseMFNConfig
    _tied_weights_keys = {}

    def __init__(self, config):
        from mfn import MyelinFatigueNet

        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.core = MyelinFatigueNet(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            output_size=config.vocab_size,
            beta=config.beta,
        )
        self.post_init()

    def forward(self, input_ids=None, state=None, labels=None,
                return_dict=None, attention_mask=None, **kwargs):
        return_dict = return_dict if return_dict is not None else True
        e = self.embed_tokens(input_ids)
        if state is not None:
            h_lam, h_psi, phi_lam, phi_psi = state
            if e.shape[1] > 1:
                e = e[:, -1:]
            h_lam, h_psi, phi_lam, phi_psi, y_t = self.core(
                e[:, 0], h_lam, h_psi, phi_lam, phi_psi)
            logits = y_t[:, None]
            return CausalLMOutput(logits=logits,
                                  hidden_states=(h_lam, h_psi, phi_lam, phi_psi))

        h_lam, h_psi, phi_lam, phi_psi = self.core.init_state(
            e.shape[0], self.config.hidden_size, e.device)
        logits = []
        for t in range(e.shape[1]):
            h_lam, h_psi, phi_lam, phi_psi, y_t = self.core(
                e[:, t], h_lam, h_psi, phi_lam, phi_psi)
            logits.append(y_t)
        logits = torch.stack(logits, dim=1)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

        if not return_dict:
            return ((loss,) if loss is not None else tuple()) + (logits,)

        return CausalLMOutput(loss=loss, logits=logits,
                              hidden_states=(h_lam, h_psi, phi_lam, phi_psi))


# ---------------------------------------------------------------- plumbing ----

def true_param_count(model):
    """Deduplicated parameter count (shared/tied weights counted once)."""
    seen = set()
    total = 0
    for name, p in model.named_parameters():
        key = id(p)
        base = name.split(".")[0]
        # tied lm_head.weight shares the same tensor as embed_tokens.weight
        if base.startswith("lm_head") and id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def build_lm(model_id: str, vocab_size: int, hidden_size: int,
             beta: float = 0.2, no_bidir: bool = False):
    """Create a model instance from a short id (matching train_lm.py --models)."""
    if model_id == "mfndiag":
        cfg = MFNDiagConfig(vocab_size=vocab_size, hidden_size=hidden_size,
                            beta=beta, no_bidir=False)
        return MFNDiagForCausalLM(cfg)
    if model_id == "mfndiag_nobidir":
        cfg = MFNDiagConfig(vocab_size=vocab_size, hidden_size=hidden_size,
                            beta=beta, no_bidir=True)
        return MFNDiagForCausalLM(cfg)
    if model_id == "gru":
        cfg = GRULMConfig(vocab_size=vocab_size, hidden_size=hidden_size)
        return GRUForCausalLM(cfg)
    if model_id == "dense_mfn":
        cfg = DenseMFNConfig(vocab_size=vocab_size, hidden_size=hidden_size,
                            beta=beta, no_bidir=False)
        return DenseMFNForCausalLM(cfg)
    raise ValueError(f"unknown model id: {model_id}")


def matched_gru_hidden(mfn_params: int, vocab_size: int) -> int:
    """Smallest H such that a GRU (tied embeddings) reaches <= mfn_params."""
    for H in range(1, 1024):
        p = vocab_size * H + 3 * (2 * H * H + 2 * H)
        if p >= mfn_params:
            return H
    raise ValueError("no H found")


def register_auto():
    AutoConfig.register(MFNDiagConfig.model_type, MFNDiagConfig)
    AutoConfig.register(GRULMConfig.model_type, GRULMConfig)
    AutoConfig.register(DenseMFNConfig.model_type, DenseMFNConfig)
    AutoModelForCausalLM.register(MFNDiagConfig, MFNDiagForCausalLM)
    AutoModelForCausalLM.register(GRULMConfig, GRUForCausalLM)
    AutoModelForCausalLM.register(DenseMFNConfig, DenseMFNForCausalLM)


register_auto()
MFNDiagConfig.register_for_auto_class("AutoConfig")
GRULMConfig.register_for_auto_class("AutoConfig")
MFNDiagForCausalLM.register_for_auto_class("AutoModelForCausalLM")
GRUForCausalLM.register_for_auto_class("AutoModelForCausalLM")