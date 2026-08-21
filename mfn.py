import torch
import torch.nn as nn


class MyelinFatigueNet(nn.Module):
    """Dual-stream recurrent architecture with bidirectional gating
    and neural fatigue regularization.

    Args:
        input_size  (int): Input dimension d.
        hidden_size (int): Per-stream hidden dimension H.
        output_size (int): Output dimension d_out.
        beta        (float): Slow-stream decay exponent (default 0.2).
    """

    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int, beta: float = 0.2):
        super().__init__()
        H, d = hidden_size, input_size
        self.H = H
        self.beta = beta

        # Shared input projection
        self.input_proj = nn.Linear(d, H)

        # Independent adaptive decay MLPs (d -> H -> H)
        self.decay_lam = nn.Sequential(
            nn.Linear(d, H), nn.ReLU(), nn.Linear(H, H), nn.Sigmoid())
        self.decay_psi = nn.Sequential(
            nn.Linear(d, H), nn.ReLU(), nn.Linear(H, H), nn.Sigmoid())

        # Within-stream feedback projections
        self.fb_lam = nn.Linear(H, H)
        self.fb_psi = nn.Linear(H, H)

        # Cross-stream projections (no bias)
        self.W_psi2lam = nn.Linear(H, H, bias=False)
        self.W_lam2psi = nn.Linear(H, H, bias=False)

        # Bidirectional confidence gates (input dim = 2H)
        self.gate_psi2lam = nn.Sequential(
            nn.Linear(2 * H, H), nn.Sigmoid())    # Psi -> Lambda
        self.gate_lam2psi = nn.Sequential(
            nn.Linear(2 * H, H), nn.Sigmoid())    # Lambda -> Psi

        # GRU update cells
        self.gru_lam = nn.GRUCell(H, H)
        self.gru_psi = nn.GRUCell(H, H)

        # Per-neuron fatigue recovery rates (raw logits; sigmoid at runtime)
        self.w_gamma_lam = nn.Parameter(torch.zeros(H))
        self.w_gamma_psi = nn.Parameter(torch.zeros(H))

        # Readout (concatenated streams -> output)
        self.readout = nn.Linear(2 * H, output_size)

    def forward(self, x, h_lam, h_psi, phi_lam, phi_psi):
        # Recovery rates (per neuron, enforced in (0,1))
        gamma_lam = torch.sigmoid(self.w_gamma_lam)      # (H,)
        gamma_psi = torch.sigmoid(self.w_gamma_psi)      # (H,)

        # Fatigue-modulated effective states (prior fatigue)
        h_lam_eff = h_lam * (1.0 - phi_lam)              # (B, H)
        h_psi_eff = h_psi * (1.0 - phi_psi)              # (B, H)

        # Input projection
        xp = self.input_proj(x)                          # (B, H)

        # Adaptive decay
        alpha_lam = self.decay_lam(x)                    # (B, H) in (0,1)
        alpha_psi = self.decay_psi(x) ** self.beta       # (B, H) slower

        # Bidirectional confidence gates (reversed argument order)
        joint_p2l = torch.cat([h_psi_eff, h_lam_eff], dim=-1)     # (B, 2H)
        joint_l2p = torch.cat([h_lam_eff, h_psi_eff], dim=-1)     # (B, 2H)
        g_p2l = self.gate_psi2lam(joint_p2l)             # (B, H)
        g_l2p = self.gate_lam2psi(joint_l2p)             # (B, H)

        # Within-stream feedback (gated by the OTHER stream)
        fb_lam = self.fb_lam(h_lam_eff) * alpha_lam * g_p2l        # (B, H)
        fb_psi = self.fb_psi(h_psi_eff) * alpha_psi * g_l2p        # (B, H)

        # Stream updates: Lambda FIRST, then Psi uses fresh h_lam
        h_lam = self.gru_lam(
            xp + fb_lam + self.W_psi2lam(h_psi_eff), h_lam)        # (B, H)

        # Tentative effective Lambda for Psi input (uses PRIOR phi_lam)
        h_lam_eff_star = h_lam * (1.0 - phi_lam)                   # (B, H)

        h_psi = self.gru_psi(
            xp + fb_psi + self.W_lam2psi(h_lam_eff_star), h_psi)   # (B, H)

        # Fatigue update (post-GRU, affects next step only)
        phi_lam = gamma_lam * phi_lam + (1.0 - gamma_lam) * h_lam.abs()
        phi_psi = gamma_psi * phi_psi + (1.0 - gamma_psi) * h_psi.abs()

        # Final effective states and readout
        h_lam_out = h_lam * (1.0 - phi_lam)
        h_psi_out = h_psi * (1.0 - phi_psi)
        y = self.readout(torch.cat([h_lam_out, h_psi_out], dim=-1))

        return h_lam, h_psi, phi_lam, phi_psi, y

    @staticmethod
    def init_state(batch_size, hidden_size, device='cpu'):
        zeros = lambda: torch.zeros(batch_size, hidden_size, device=device)
        return zeros(), zeros(), zeros(), zeros()


def run_sequence(model, x_seq, batch_size, hidden_size, device='cpu'):
    h_lam, h_psi, phi_lam, phi_psi = MyelinFatigueNet.init_state(
        batch_size, hidden_size, device)
    outs = []
    fatigue_lam, fatigue_psi = [], []
    for x_t in x_seq:
        h_lam, h_psi, phi_lam, phi_psi, y_t = \
            model(x_t, h_lam, h_psi, phi_lam, phi_psi)
        outs.append(y_t)
        fatigue_lam.append(phi_lam.detach().mean().item())
        fatigue_psi.append(phi_psi.detach().mean().item())
    return torch.stack(outs), fatigue_lam, fatigue_psi


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())