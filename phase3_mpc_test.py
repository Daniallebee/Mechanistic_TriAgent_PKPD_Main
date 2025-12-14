# --------------------------------------------------------------
# phase3_mpc_test.py — TEST MULTI-STEP MPC
# --------------------------------------------------------------

from triagent_controller.pas_interface import PASInterface
from triagent_controller.mpc_controller import MPCController
from triagent_controller.state_action import action_to_infusions   # <-- FIXED

def run_mpc_sim():

    print("\n===== PHASE 3 — MULTI-STEP MPC (20s horizon) =====\n")

    env = PASInterface(dt=1.0)
    obs = env.reset()

    print("Initial State:", obs, "\n")

    mpc = MPCController(horizon=20)

    # Run 60 seconds of closed-loop control
    for t in range(60):

        action_id = mpc.choose_action(env)
        inf = action_to_infusions(action_id)
        obs = env.step(inf)

        print(
            f"t={t:02d}s | action={action_id:02d} "
            f"| BIS={obs['BIS']:.1f}  MAP={obs['MAP']:.1f}  HR={obs['HR']:.1f}"
        )

    print("\n===== MPC SIMULATION COMPLETE =====\n")


if __name__ == "__main__":
    run_mpc_sim()
