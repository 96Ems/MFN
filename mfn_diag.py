import torch
import torch.nn as nn


class MyelinFatigueNetDiag(nn.Module):
    """O(H) variant of MFN. Every dense H x H (or 2H x H) interaction from the
    original architecture is replaced by an elementwise (diagonal) operation:

      - GRUCell (12H^2) is DROPPED entirely. The existing adaptive-decay alpha
        (already input-conditioned, per-neuron, in (0,1)^H -- functionally
        analogous to Mamba's Delta, as the paper itself notes in Sec 2.5) now
        performs the recurrence directly:
            h_t = alpha_t * h_{t-1} + (1 - alpha_t) * tanh(mix_t)
      - Cross-stream projections W_psi2lam, W_lam2psi (2H^2) -> diagonal vectors.
      - Confidence gates (4H^2, dense over concatenated 2H input) -> two
        diagonal vectors each (one per source stream) instead of one dense
        2H x H matrix.
      - Feedback projections W_fb (2H^2) -> diagonal vectors.
      - Decay MLPs' second dense layer (2H^2) -> elementwise scale + bias.

    Input projection and readout stay dense (they scale with d and d_out, not
    H^2, so they don't affect the asymptotic per-step cost).
    """

    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int, beta: float = 0.2):
        super().__init__()
        H, d = hidden_size, input_size
        self.H = H
        self.beta = beta

        self.input_proj = nn.Linear(d, H)

        # Decay: dense d->H first layer (O(dH), not H^2), then diagonal scale+bias
        self.decay_lam_in = nn.Linear(d, H)
        self.decay_lam_scale = nn.Parameter(torch.ones(H))
        self.decay_lam_bias = nn.Parameter(torch.zeros(H))
        self.decay_psi_in = nn.Linear(d, H)
        self.decay_psi_scale = nn.Parameter(torch.ones(H))
        self.decay_psi_bias = nn.Parameter(torch.zeros(H))

        # Feedback: diagonal instead of dense H x H
        self.fb_lam_diag = nn.Parameter(torch.randn(H) * 0.1)
        self.fb_psi_diag = nn.Parameter(torch.randn(H) * 0.1)

        # Cross-stream: diagonal instead of dense H x H
        self.W_psi2lam_diag = nn.Parameter(torch.randn(H) * 0.1)
        self.W_lam2psi_diag = nn.Parameter(torch.randn(H) * 0.1)

        # Bidirectional gates: diagonal on each source instead of dense 2H->H
        self.gate_p2l_u = nn.Parameter(torch.randn(H) * 0.1)  # weight on h_psi
        self.gate_p2l_v = nn.Parameter(torch.randn(H) * 0.1)  # weight on h_lam
        self.gate_p2l_b = nn.Parameter(torch.zeros(H))
        self.gate_l2p_u = nn.Parameter(torch.randn(H) * 0.1)  # weight on h_lam
        self.gate_l2p_v = nn.Parameter(torch.randn(H) * 0.1)  # weight on h_psi
        self.gate_l2p_b = nn.Parameter(torch.zeros(H))

        self.w_gamma_lam = nn.Parameter(torch.zeros(H))
        self.w_gamma_psi = nn.Parameter(torch.zeros(H))

        self.readout = nn.Linear(2 * H, output_size)

    def forward(self, x, h_lam, h_psi, phi_lam, phi_psi):
        gamma_lam = torch.sigmoid(self.w_gamma_lam)
        gamma_psi = torch.sigmoid(self.w_gamma_psi)

        h_lam_eff = h_lam * (1.0 - phi_lam)
        h_psi_eff = h_psi * (1.0 - phi_psi)

        xp = self.input_proj(x)

        alpha_lam = torch.sigmoid(self.decay_lam_scale * self.decay_lam_in(x)
                                   + self.decay_lam_bias)
        alpha_psi = torch.sigmoid(self.decay_psi_scale * self.decay_psi_in(x)
                                   + self.decay_psi_bias) ** self.beta

        g_p2l = torch.sigmoid(self.gate_p2l_u * h_psi_eff
                               + self.gate_p2l_v * h_lam_eff + self.gate_p2l_b)
        g_l2p = torch.sigmoid(self.gate_l2p_u * h_lam_eff
                               + self.gate_l2p_v * h_psi_eff + self.gate_l2p_b)

        fb_lam = self.fb_lam_diag * h_lam_eff * alpha_lam * g_p2l
        fb_psi = self.fb_psi_diag * h_psi_eff * alpha_psi * g_l2p

        mix_lam = xp + fb_lam + self.W_psi2lam_diag * h_psi_eff
        h_lam = alpha_lam * h_lam + (1.0 - alpha_lam) * torch.tanh(mix_lam)

        h_lam_eff_star = h_lam * (1.0 - phi_lam)

        mix_psi = xp + fb_psi + self.W_lam2psi_diag * h_lam_eff_star
        h_psi = alpha_psi * h_psi + (1.0 - alpha_psi) * torch.tanh(mix_psi)

        phi_lam = gamma_lam * phi_lam + (1.0 - gamma_lam) * h_lam.abs()
        phi_psi = gamma_psi * phi_psi + (1.0 - gamma_psi) * h_psi.abs()

        h_lam_out = h_lam * (1.0 - phi_lam)
        h_psi_out = h_psi * (1.0 - phi_psi)
        y = self.readout(torch.cat([h_lam_out, h_psi_out], dim=-1))

        return h_lam, h_psi, phi_lam, phi_psi, y

    @staticmethod
    def init_state(batch_size, hidden_size, device='cpu'):
        zeros = lambda: torch.zeros(batch_size, hidden_size, device=device)
        return zeros(), zeros(), zeros(), zeros()


def run_sequence_diag(model, x_seq, batch_size, hidden_size, device='cpu'):
    h_lam, h_psi, phi_lam, phi_psi = MyelinFatigueNetDiag.init_state(
        batch_size, hidden_size, device)
    outs = []
    for x_t in x_seq:
        h_lam, h_psi, phi_lam, phi_psi, y_t = model(x_t, h_lam, h_psi, phi_lam, phi_psi)
        outs.append(y_t)
    return torch.stack(outs)
