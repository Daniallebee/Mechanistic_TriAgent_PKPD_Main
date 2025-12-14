# --------------------------------------------------------------
# plot_phase3_B_action_costs.py
# ACTION COSTS FOR ALL 27 ACTIONS
# --------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

from triagent_controller.pas_interface import PASInterface
from triagent_controller.mpc_controller import MPCController
from triagent_controller.state_action import NUM_ACTIONS

ROOT = "plots_phase3"
os.makedirs(ROOT, exist_ok=True)

H = 20
W_BIS = 2.0
W_MAP = 1.0
W_ACT = 0.01

# --------------------------------------------------------------
# Compute cost J(a) for each action at t=0
# --------------------------------------------------------------
def evaluate_action_costs():
    env = PASInterface(dt=1.0)
    env.reset()

    mpc = MPCController(
        w_bis=W_BIS,
        w_map=W_MAP,
        w_act=W_ACT,
        horizon=H
    )

    costs = np.zeros(NUM_ACTIONS)

    for a in range(NUM_ACTIONS):
        costs[a] = mpc.evaluate_action(env, a)

    return costs


# --------------------------------------------------------------
# PLOT
# --------------------------------------------------------------
if __name__ == "__main__":
    costs = evaluate_action_costs()

    plt.figure(figsize=(12,5))
    plt.bar(np.arange(NUM_ACTIONS), costs, color="gray")
    plt.title("Phase 3 — MPC Action Costs (t=0)")
    plt.xlabel("Action ID")
    plt.ylabel("Objective Cost J(a)")
    plt.grid(axis="y")

    plt.savefig(os.path.join(ROOT, "P3B_action_costs.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved Phase 3 Figure B → plots_phase3/P3B_action_costs.png")
