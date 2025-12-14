# --------------------------------------------------------------
# plot_phase3_C_predictions.py
# MPC 20-STEP PREDICTION HORIZON (initial state)
# --------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

from triagent_controller.pas_interface import PASInterface
from triagent_controller.mpc_controller import MPCController

ROOT = "plots_phase3"
os.makedirs(ROOT, exist_ok=True)

H = 20
W_BIS = 2.0
W_MAP = 1.0
W_ACT = 0.01

if __name__ == "__main__":

    env = PASInterface(dt=1.0)
    env.reset()

    mpc = MPCController(
        w_bis=W_BIS,
        w_map=W_MAP,
        w_act=W_ACT,
        horizon=H
    )

    # ----------------------------------------------------------
    # Get predicted trajectories
    # ----------------------------------------------------------
    bis_pred, map_pred = mpc.predict_trajectory(env, horizon=H)

    t = np.arange(H)

    plt.figure(figsize=(12,5))
    plt.plot(t, bis_pred, label="Predicted BIS", color="blue", linewidth=2)
    plt.plot(t, map_pred, label="Predicted MAP", color="red", linewidth=2)
    plt.title("Phase 3 — MPC 20-Step Prediction Horizon")
    plt.xlabel("Prediction Step")
    plt.ylabel("Predicted Values")
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(ROOT, "P3C_prediction_horizon.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved Phase 3 Figure C → plots_phase3/P3C_prediction_horizon.png")
