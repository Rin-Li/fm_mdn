from pfp.common.se3_utils import init_random_traj_th
from pfp.common.visualization import RerunViewer as RV

B = 2
T = 10
traj = init_random_traj_th(B, T)
# Vis first traj
RV("random_traj")
RV.add_traj("random_traj", traj[0].cpu().numpy())

print("Done.")
