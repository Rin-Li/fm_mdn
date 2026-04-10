import torch
import trimesh
import numpy as np
import pypose as pp
import rerun as rr


def rr_init(name: str, addr: str = None):
    rr.init(name)
    if addr is None:
        addr = "127.0.0.1"
    port = ":9876"
    rr.connect(addr + port)
    rr.log("vis", rr.Clear(recursive=True))
    return


def rr_add_axis(name: str, pose: np.ndarray, size: float = 0.004, timeless: bool = False):
    mesh = trimesh.creation.axis(origin_size=size, transform=pose)
    # Handle colors
    if mesh.visual.kind in ["vertex", "face"]:
        vertex_colors = mesh.visual.vertex_colors
    elif mesh.visual.kind == "texture":
        vertex_colors = mesh.visual.to_color().vertex_colors
    else:
        vertex_colors = None
    # Log mesh
    rr_mesh = rr.Mesh3D(
        vertex_positions=mesh.vertices,
        vertex_colors=vertex_colors,
        vertex_normals=mesh.vertex_normals,
        indices=mesh.faces,
    )
    rr.log(name, rr_mesh, timeless=timeless)
    return


# Print random twists
for _ in range(50):
    euler = torch.FloatTensor(1, 3).uniform_(-torch.pi / 2, torch.pi / 2)
    twist = pp.Log(pp.euler2SO3(euler))
    # print(euler)
    # print(twist)


# Exp-map: se3 -> SE3
# Identity (SE3) + twist (se3) = pose (SE3)
twist = pp.randn_se3(1)
pose = pp.Exp(twist)
del twist, pose

# Log-map: SE3 -> se3
# pose (SE3) - Identity (SE3) = twist (se3)
pose = pp.randn_SE3(1)
twist = pp.Log(pose)
del pose, twist

# Right-plus operator: SE3 + se3 = SE3
# This is in local frame, i.e. in the tangent space of the start pose
start_pose = pp.randn_SE3(1)
twist = pp.randn_se3(1)
end_pose = start_pose @ pp.Exp(twist)
del start_pose, twist, end_pose

# Right-minus operator: SE3 - SE3 = se3
# This is in local frame, i.e. in the tangent space of the start pose
start_pose = pp.randn_SE3(1)
end_pose = pp.randn_SE3(1)
twist = pp.Log(pp.Inv(start_pose) @ end_pose)
del start_pose, end_pose, twist

# Left-plus operator: SE3 + se3 = SE3
# This is in global frame, i.e. in the tangent space of the identity pose
start_pose = pp.randn_SE3(1)
twist = pp.randn_se3(1)
end_pose_a = start_pose + twist
end_pose_b = pp.Retr(start_pose, twist)
end_pose_c = pp.Exp(twist) @ start_pose
assert torch.allclose(end_pose_a, end_pose_b)
assert torch.allclose(end_pose_a, end_pose_c)
del start_pose, twist, end_pose_a, end_pose_b, end_pose_c

# Left-minus operator: SE3 - SE3 = se3
# This is in global frame, i.e. in the tangent space of the identity pose
start_pose = pp.randn_SE3(1)
end_pose = pp.randn_SE3(1)
twist = pp.Log(end_pose @ pp.Inv(start_pose))
del start_pose, end_pose, twist

# Simple simulation
rr_init("sandbox")
T0_th = torch.eye(4).unsqueeze(0)
T0_th[:, :3, 3] = torch.tensor([0.5, 0, 0])
# Rotate -90 degrees
T0_th[:, :3, 0] = torch.tensor([0, -1, 0])
T0_th[:, :3, 1] = torch.tensor([1, 0, 0])

T1_th = torch.eye(4).unsqueeze(0)
T1_th[:, :3, 3] = torch.tensor([0, 0.5, 0])
T1_th[:, :3, 0], T1_th[:, :3, 1] = T1_th[:, :3, 1], -T1_th[:, :3, 0]  # Rotate 90 degrees

rr_add_axis("vis/origin", np.eye(4), size=0.01, timeless=True)
rr_add_axis("vis/T0", T0_th[0].numpy(), timeless=True)
rr_add_axis("vis/T1", T1_th[0].numpy(), timeless=True)


T0_pp = pp.mat2SE3(T0_th)
T1_pp = pp.mat2SE3(T1_th)
vel_right = pp.Log(pp.Inv(T0_pp) @ T1_pp)
vel_left = pp.Log(T1_pp @ pp.Inv(T0_pp))

k_steps = 50
dt = 1 / k_steps
pose_pp_right = T0_pp
pose_pp_left = T0_pp
for k_step in range(k_steps):
    pose_pp_right: pp.LieTensor = pose_pp_right @ pp.Exp(vel_right * dt)
    pose_pp_left: pp.LieTensor = pp.Exp(vel_left * dt) @ pose_pp_left
    rr.set_time_sequence("k_step", k_step)
    rr_add_axis("vis/pose_se3_right", pose_pp_right.matrix().squeeze().numpy())
    rr_add_axis("vis/pose_se3_left", pose_pp_left.matrix().squeeze().numpy())

# SO3 x R3
t0 = T0_th[0, :3, 3]
t1 = T1_th[0, :3, 3]
R0_pp = pp.mat2SO3(T0_th[:, :3, :3])
R1_pp = pp.euler2SO3(torch.tensor([0, 0, np.pi / 2 - 0.001]))
xyz_vel = t1 - t0
R_vel_right = pp.Log(pp.Inv(R0_pp) @ R1_pp)
R_vel_left = pp.Log(R1_pp @ pp.Inv(R0_pp))

xyz = t0
R_pp_right = R0_pp
R_pp_left = R0_pp
for k_step in range(k_steps):
    xyz = xyz + xyz_vel * dt
    R_pp_right = R_pp_right @ pp.Exp(R_vel_right * dt)
    R_pp_left = pp.Exp(R_vel_left * dt) @ R_pp_left
    rr.set_time_sequence("k_step", k_step)
    pose_right = torch.eye(4)
    pose_right[:3, :3] = R_pp_right.matrix().squeeze()
    pose_right[:3, 3] = xyz
    pose_left = torch.eye(4)
    pose_left[:3, :3] = R_pp_left.matrix().squeeze()
    pose_left[:3, 3] = xyz
    rr_add_axis("vis/pose_so3", pose_right.numpy())
    rr_add_axis("vis/pose_so3_left", pose_left.numpy())

print(f"{R_vel_right=}")
print(f"{R_vel_left=}")
print("done")
