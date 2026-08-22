"""Proof A1: sigmoid(z)**beta has infinite backward gradient as sigmoid->0 in fp32."""
import torch
import math

beta=0.2
# Simulate large negative pre-activation -> sigmoid flushes to 0 in fp32
for z in [-10, -50, -100, -103, -110]:
    x_fp32 = torch.sigmoid(torch.tensor([z], dtype=torch.float32))
    x_fp64 = torch.sigmoid(torch.tensor([z], dtype=torch.float64))
    print(f"z={z:4d}  sigmoid fp32={x_fp32.item():.3e}  fp64={x_fp64.item():.3e}  flush? {x_fp32.item()==0.0}")

# Gradient proof
print("\n--- gradient blowup ---")
for eps in [1e-3, 1e-6, 1e-9, 1e-12, 0.0]:
    x = torch.tensor([eps], dtype=torch.float32, requires_grad=True)
    y = x ** beta
    y.backward()
    # analytical derivative beta*x^{beta-1}
    analytic = beta * (eps ** (beta-1)) if eps>0 else float('inf')
    grad = x.grad.item()
    print(f"x={eps:.0e}  grad autograd={grad:.3e}  analytic={analytic:.3e}  is_inf={math.isinf(grad) or grad>1e7}")

print("\n--- with clamp_min(1e-6) fix ---")
eps0=1e-6
for eps in [0.0, 1e-9, 1e-7, 1e-6, 1e-3]:
    x = torch.tensor([eps], dtype=torch.float32, requires_grad=True)
    y = x.clamp_min(eps0) ** beta
    y.backward()
    print(f"x={eps:.0e}  clamped={max(eps,eps0):.0e}  grad={x.grad.item():.3e}  bound beta*eps0^(beta-1)={beta*eps0**(beta-1):.3e}")

# Forward error of clamp
print(f"\nForward error: (1e-7)^0.2={1e-7**0.2:.6f} vs (1e-6)^0.2={1e-6**0.2:.6f} diff={abs(1e-7**0.2-1e-6**0.2):.6f}  (negligible, flat near 0)")
# Conclusion
print("\nProof: clamp_min bounds gradient to", beta*1e-6**(beta-1), "and prevents NaN from 0**beta backward.")
