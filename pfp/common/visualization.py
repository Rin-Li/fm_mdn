from __future__ import annotations
import trimesh
import numpy as np
import open3d as o3d
from yourdfpy.urdf import URDF
from pfp.common.se3_utils import pfp_to_pose_np

try:
    import rerun as rr
except ImportError:
    print("WARNING: Rerun not installed. Visualization will not work.")


class RerunViewer:
    def __init__(self, name: str, addr: str = None):
        rr.init(name)
        if addr is None:
            addr = "127.0.0.1"
        port = ":9876"
        rr.connect(addr + port)
        RerunViewer.clear()
        return

    @staticmethod
    def add_obs_dict(obs_dict: dict, timestep: int = None):
        if timestep is not None:
            rr.set_time_sequence("timestep", timestep)
        RerunViewer.add_rgb("rgb", obs_dict["image"])
        RerunViewer.add_depth("depth", obs_dict["depth"])
        RerunViewer.add_np_pointcloud(
            "vis/pointcloud",
            points=obs_dict["point_cloud"][:, :3],
            colors_uint8=obs_dict["point_cloud"][:, 3:],
        )
        return

    @staticmethod
    def add_o3d_pointcloud(name: str, pointcloud: o3d.geometry.PointCloud, radii: float = None):
        points = np.asanyarray(pointcloud.points)
        colors = np.asanyarray(pointcloud.colors) if pointcloud.has_colors() else None
        colors_uint8 = (colors * 255).astype(np.uint8) if pointcloud.has_colors() else None
        RerunViewer.add_np_pointcloud(name, points, colors_uint8, radii)
        return

    @staticmethod
    def add_np_pointcloud(
        name: str, points: np.ndarray, colors_uint8: np.ndarray = None, radii: float = None
    ):
        rr_points = rr.Points3D(positions=points, colors=colors_uint8, radii=radii)
        rr.log(name, rr_points)
        return

    @staticmethod
    def add_axis(name: str, pose: np.ndarray, size: float = 0.004, timeless: bool = False):
        mesh = trimesh.creation.axis(origin_size=size, transform=pose)
        RerunViewer.add_mesh_trimesh(name, mesh, timeless)
        return

    @staticmethod
    def add_aabb(name: str, centers: np.ndarray, extents: np.ndarray, timeless=False):
        rr.log(name, rr.Boxes3D(centers=centers, sizes=extents), timeless=timeless)
        return

    @staticmethod
    def add_mesh_trimesh(name: str, mesh: trimesh.Trimesh, timeless: bool = False):
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

    @staticmethod
    def add_mesh_list_trimesh(name: str, meshes: list[trimesh.Trimesh]):
        for i, mesh in enumerate(meshes):
            RerunViewer.add_mesh_trimesh(name + f"/{i}", mesh)
        return

    @staticmethod
    def add_rgb(name: str, rgb_uint8: np.ndarray):
        if rgb_uint8.shape[0] == 3:
            # CHW -> HWC
            rgb_uint8 = np.transpose(rgb_uint8, (1, 2, 0))
        rr.log(name, rr.Image(rgb_uint8))

    @staticmethod
    def add_depth(name: str, detph: np.ndarray):
        rr.log(name, rr.DepthImage(detph))

    @staticmethod
    def add_traj(name: str, traj: np.ndarray):
        """
        name: str
        traj: np.ndarray (T, 10)
        """
        poses = pfp_to_pose_np(traj)
        for i, pose in enumerate(poses):
            RerunViewer.add_axis(name + f"/{i}t", pose)
        return

    @staticmethod
    def clear():
        rr.log("vis", rr.Clear(recursive=True))
        return


class RerunTraj:
    def __init__(self) -> None:
        self.traj_shape = None
        return

    def add_traj(self, name: str, traj: np.ndarray, size: float = 0.004):
        """
        name: str
        traj: np.ndarray (T, 10)
        """
        if self.traj_shape is None or self.traj_shape != traj.shape:
            self.traj_shape = traj.shape
            for i in range(traj.shape[0]):
                RerunViewer.add_axis(name + f"/{i}t", np.eye(4), size)
        poses = pfp_to_pose_np(traj)
        for i, pose in enumerate(poses):
            rr.log(
                name + f"/{i}t",
                rr.Transform3D(mat3x3=pose[:3, :3], translation=pose[:3, 3]),
            )
        return


class RerunURDF:
    def __init__(self, name: str, urdf_path: str, meshes_root: str):
        self.name = name
        self.urdf: URDF = URDF.load(urdf_path, mesh_dir=meshes_root)
        return

    def update_vis(
        self,
        joint_state: list | np.ndarray,
        root_pose: np.ndarray = np.eye(4),
        name_suffix: str = "",
    ):
        self._update_joints(joint_state)
        scene = self.urdf.scene
        trimeshes = self._scene_to_trimeshes(scene)
        trimeshes = [t.apply_transform(root_pose) for t in trimeshes]
        RerunViewer.add_mesh_list_trimesh(self.name + name_suffix, trimeshes)
        return

    def _update_joints(self, joint_state: list | np.ndarray):
        assert len(joint_state) == len(self.urdf.actuated_joints), "Wrong number of joint values."
        self.urdf.update_cfg(joint_state)
        return

    def _scene_to_trimeshes(self, scene: trimesh.Scene) -> list[trimesh.Trimesh]:
        """
        Convert a trimesh.Scene to a list of trimesh.Trimesh.

        Skips objects that are not an instance of trimesh.Trimesh.
        """
        trimeshes = []
        scene_dump = scene.dump()
        geometries = [scene_dump] if not isinstance(scene_dump, list) else scene_dump
        for geometry in geometries:
            if isinstance(geometry, trimesh.Trimesh):
                trimeshes.append(geometry)
            elif isinstance(geometry, trimesh.Scene):
                trimeshes.extend(self._scene_to_trimeshes(geometry))
        return trimeshes
