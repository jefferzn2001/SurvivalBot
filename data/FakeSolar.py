#!/usr/bin/env python3
"""
Solar-panel current vs. ALS-PT19 reading
Power-law fit through:
  • (240, 0   A)   darkness cut-off
  • (246, 0.4 A)   first noticeable current
  • (941, 2.34 A)  panel’s max ≈ 30 W / 12.8 V
"""

import numpy as np
import matplotlib.pyplot as plt

# --- constants ---
X0       = 240            # cut-off (ADC counts)
K_FACTOR = 0.2057465367   # fitted scale factor
N_EXP    = 0.37104285     # fitted exponent
I_MAX    = 2.34           # clamp (A)
ADC_MAX  = 1023           # 10-bit ADC full-scale

def panel_current(adc: float) -> float:
    """Return simulated panel current (A) for a given ADC reading."""
    if adc <= X0:
        return 0.0
    amps = K_FACTOR * (adc - X0) ** N_EXP
    return min(amps, I_MAX)          # clamp to 2.34 A

if __name__ == "__main__":
    xs = np.linspace(0, ADC_MAX, 1024)
    ys = [panel_current(x) for x in xs]

    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, linewidth=2)
    plt.xlabel("Average LDR reading")
    plt.ylabel("Panel current (A)")
    plt.title("Fake Solar Panel Current based on LDR")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
