# --------------------------------------------------------------
# plot_phase3_A_control_trajectory.py
# MPC CONTROL TRAJECTORY — 60s
# --------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

from triagent_controller.pas_interface import PASInterface
from triagent_controller.mpc_controller import MPCController
from triagent_controller.state_action import action_to_infusions

# --------------------------------------------------------------
# Directory
# --------------------------------------------------------------
ROOT = "plots_phase3"
os.makedirs(ROOT, exist_ok=True)

# --------------------------------------------------------------
# MPC parameters
# --------------------------------------------------------------
H = 20     # MPC prediction horizon (seconds)
W_BIS = 2.0
W_MAP = 1.0
W_ACT = 0.01

# --------------------------------------------------------------
# RUN 60-second MPC SIMULATION
# --------------------------------------------------------------
def run_mpc_sim():
    env = PASInterface(dt=1.0)
    mpc = MPCController(
        w_bis=W_BIS,
        w_map=W_MAP,
        w_act=W_ACT,
        horizon=H
    )

    obs = env.reset()

    T = 60
    BIS = []
    MAP = []
    ACTIONS = []

    for t in range(T):
        action_id = mpc.choose_action(env)
        ACTIONS.append(action_id)

        obs = env.step(action_to_infusions(action_id))

        BIS.append(obs["BIS"])
        MAP.append(obs["MAP"])

    return np.array(BIS), np.array(MAP), np.array(ACTIONS)


# --------------------------------------------------------------
# PLOT
# --------------------------------------------------------------
if __name__ == "__main__":
    BIS, MAP, ACT = run_mpc_sim()

    t = np.arange(60)

    fig, ax1 = plt.subplots(figsize=(12,6))

    ax1.plot(t, BIS, label="BIS", color="blue", linewidth=2)
    ax1.set_ylabel("BIS", color="blue")
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    ax2.plot(t, MAP, label="MAP", color="red", linewidth=2)
    ax2.set_ylabel("MAP (mmHg)", color="red")
    ax2.set_ylim(40, 120)

    plt.title("Phase 3 — MPC Control Trajectory (60s)")
    plt.grid(True)

    # Action subplot
    plt.figure(figsize=(12,3))
    plt.step(t, ACT, where="post", color="black")
    plt.title("MPC — Action Selected Over Time")
    plt.ylabel("Action ID")
    plt.xlabel("Time (s)")
    plt.grid(True)

    plt.savefig(os.path.join(ROOT, "P3A_control_trajectory.png"), dpi=300, bbox_inches="tight")
    plt.close("all")

    print("Saved Phase 3 Figure A → plots_phase3/P3A_control_trajectory.png")
