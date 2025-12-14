# --------------------------------------------------------------
# mpc_controller.py — FINAL MULTI-STEP MPC (Phase 3)
# --------------------------------------------------------------

import numpy as np
from triagent_controller.state_action import NUM_ACTIONS, action_to_infusions


class MPCController:

    def __init__(self, w_bis=1.0, w_map=1.0, w_act=0.0, horizon=20):
        """
        w_bis  : BIS penalty weight
        w_map  : MAP penalty weight
        w_act  : small penalty for high action index (smoothness)
        horizon: MPC prediction depth
        """
        self.w_bis = w_bis
        self.w_map = w_map
        self.w_act = w_act
        self.horizon = horizon

    # ==========================================================
    # MPC CHOOSES ACTION WITH MINIMUM COST
    # ==========================================================
    def choose_action(self, env):
        best_a = None
        best_cost = float("inf")

        for a in range(NUM_ACTIONS):
            cost = self.evaluate_action(env, a)

            if cost < best_cost:
                best_cost = cost
                best_a = a

        return best_a

    # ==========================================================
    # Cost evaluation J(a) over horizon H
    # ==========================================================
    def evaluate_action(self, env, a):
        temp = self.clone_env(env)
        inf = action_to_infusions(a)

        BIS = []
        MAP = []

        for _ in range(self.horizon):
            obs = temp.step(inf)
            BIS.append(obs["BIS"])
            MAP.append(obs["MAP"])

        BIS = np.array(BIS)
        MAP = np.array(MAP)

        # Weighted squared error
        bis_err = np.mean((BIS - 50.0) ** 2)
        map_err = np.mean((MAP - 75.0) ** 2)

        cost = (
            self.w_bis * bis_err +
            self.w_map * map_err +
            self.w_act * a
        )

        return cost

    # ==========================================================
    # Predict only trajectory (for plotting)
    # ==========================================================
    def predict_trajectory(self, env, a=None, horizon=None):
        if horizon is None:
            horizon = self.horizon

        if a is None:
            a = self.choose_action(env)

        temp = self.clone_env(env)
        inf = action_to_infusions(a)

        BIS = []
        MAP = []

        for _ in range(horizon):
            obs = temp.step(inf)
            BIS.append(obs["BIS"])
            MAP.append(obs["MAP"])

        return np.array(BIS), np.array(MAP)

    # ==========================================================
    # Environment deep copy
    # ==========================================================
    def clone_env(self, env):
        from triagent_controller.pas_interface import PASInterface

        new_env = PASInterface(dt=env.dt, disturbance_profile=env.disturbance_profile)

        # Copy PK
        new_env.patient.propo_pk.x = env.patient.propo_pk.x.copy()
        new_env.patient.remi_pk.x  = env.patient.remi_pk.x.copy()
        new_env.patient.nore_pk.x  = env.patient.nore_pk.x.copy()

        # Copy effect-site
        new_env.patient.c_es_propo = getattr(env.patient, "c_es_propo", 0.0)
        new_env.patient.c_es_remi  = getattr(env.patient, "c_es_remi", 0.0)
        new_env.patient.c_blood_nore = getattr(env.patient, "c_blood_nore", 0.0)

        # Copy BIS
        new_env.patient.bis = env.patient.bis
        if hasattr(env.patient.bis_pd, "bis_buffer"):
            new_env.patient.bis_pd.bis_buffer = env.patient.bis_pd.bis_buffer.copy()

        # Copy hemodynamics
        new_env.patient.tpr = env.patient.tpr
        new_env.patient.sv  = env.patient.sv
        new_env.patient.hr  = env.patient.hr
        new_env.patient.map = env.patient.map
        new_env.patient.co  = env.patient.co

        if hasattr(env.patient.hemo_pd, "x_effect"):
            new_env.patient.hemo_pd.x_effect = env.patient.hemo_pd.x_effect.copy()

        # Copy time
        new_env.time = env.time

        return new_env
