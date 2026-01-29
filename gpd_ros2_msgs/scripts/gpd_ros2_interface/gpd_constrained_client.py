#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.task import Future

# ROS messages
from std_msgs.msg import Header, Int64
from geometry_msgs.msg import Vector3, Transform, Point
from sensor_msgs.msg import PointCloud2, PointField
from gpd_ros2_msgs.srv import DetectConstrainedGrasps
from gpd_ros2_msgs.msg import CloudIndexed, CloudSources, GraspParams

try:
  import open3d as o3d  # optional
  _HAS_O3D = True
except Exception:
  _HAS_O3D = False

def _to_numpy_xyz(pcd: Union[np.ndarray, 'o3d.geometry.PointCloud']) -> np.ndarray:
  if isinstance(pcd, np.ndarray):
    assert pcd.shape[1] == 3, "pcd must be Nx3"
    return pcd.astype(np.float32, copy=False)
  if _HAS_O3D and isinstance(pcd, o3d.geometry.PointCloud):
    return np.asarray(pcd.points, dtype=np.float32)
  raise TypeError("pcd must be numpy Nx3 or open3d.geometry.PointCloud")

def _pcd_bounds(xyz: np.ndarray):
  if xyz.size == 0:
    return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
  return xyz.min(axis=0), xyz.max(axis=0)

def _pad_workspace(min_xyz: np.ndarray, max_xyz: np.ndarray, margin: float) -> List[float]:
  return [
    float(min_xyz[0] - margin), float(max_xyz[0] + margin),
    float(min_xyz[1] - margin), float(max_xyz[1] + margin),
    float(min_xyz[2] - margin), float(max_xyz[2] + margin),
  ]

def _normalize(v: np.ndarray) -> np.ndarray:
  n = np.linalg.norm(v)
  if n < 1e-9:
    return v
  return v / n

def _vector3(xyz: Iterable[float]) -> Vector3:
  v = Vector3()
  v.x, v.y, v.z = [float(x) for x in xyz]
  return v

def _point(xyz: Iterable[float]) -> Point:
  p = Point()
  p.x, p.y, p.z = [float(x) for x in xyz]
  return p

def _transform_from_rt(translation_xyz: Iterable[float], quat_xyzw: Iterable[float]) -> Transform:
  t = Transform()
  tx, ty, tz = translation_xyz
  qx, qy, qz, qw = quat_xyzw
  t.translation.x = float(tx)
  t.translation.y = float(ty)
  t.translation.z = float(tz)
  t.rotation.x = float(qx)
  t.rotation.y = float(qy)
  t.rotation.z = float(qz)
  t.rotation.w = float(qw)
  return t

def _xyz_to_pc2(xyz: np.ndarray, frame_id: str, stamp) -> PointCloud2:
  msg = PointCloud2()
  msg.header.stamp = stamp
  msg.header.frame_id = frame_id
  msg.height = 1
  msg.width = xyz.shape[0]
  msg.fields = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
  ]
  msg.is_bigendian = False
  msg.point_step = 12
  msg.row_step = msg.point_step * msg.width
  msg.is_dense = True
  msg.data = np.asarray(xyz, dtype=np.float32).tobytes()
  return msg

@dataclass
class StretchApproachConfig:
  max_lift_z: float = 1.1
  table_clearance: float = 0.05
  fallback_dir: Tuple[float, float, float] = (1.0, 0.0, 0.0)

def decide_approach_direction(object_xyz: np.ndarray,
                              cfg: StretchApproachConfig) -> np.ndarray:
  if object_xyz.size == 0:
    return np.array([0.0, 0.0, -1.0], dtype=np.float32)
  _, maxs = _pcd_bounds(object_xyz)
  top = float(maxs[2])
  if top + cfg.table_clearance < cfg.max_lift_z:
    return np.array([0.0, 0.0, -1.0], dtype=np.float32)
  return np.array(cfg.fallback_dir, dtype=np.float32)

class GpdConstrainedClient(Node):
  def __init__(self):
    super().__init__("gpd_constrained_client")
    self.cli = self.create_client(DetectConstrainedGrasps, "detect_constrained_grasps")
    if not self.cli.wait_for_service(timeout_sec=2.0):
      self.get_logger().warn("detect_constrained_grasps service not immediately available. Will still try to call.")

  def call_with_pcds(
      self,
      pcd_obj: Union[np.ndarray, 'o3d.geometry.PointCloud'],
      pcd_env: Union[np.ndarray, 'o3d.geometry.PointCloud'],
      frame_id: str,
      cam_positions: List[Tuple[float, float, float]],
      cam_to_base: Tuple[float, float, float, float, float, float, float],
      params_policy: int,
      approach_threshold_deg: float = 20.0,
      workspace_margin: float = 0.02,
      enable_approach_filter: bool = True,
      stretch_max_lift_z: float = 1.1,
  ) -> DetectConstrainedGrasps.Response:

    obj_xyz = _to_numpy_xyz(pcd_obj)
    env_xyz = _to_numpy_xyz(pcd_env)

    full_xyz = np.vstack([env_xyz, obj_xyz]) if env_xyz.size and obj_xyz.size else (
      obj_xyz if env_xyz.size == 0 else env_xyz
    )
    obj_start = env_xyz.shape[0]
    obj_count = obj_xyz.shape[0]
    obj_indices = list(range(obj_start, obj_start + obj_count))

    stamp = self.get_clock().now().to_msg()
    cloud_msg = _xyz_to_pc2(full_xyz, frame_id, stamp)

    sources = CloudSources()
    sources.cloud = cloud_msg
    for (cx, cy, cz) in cam_positions:
        sources.view_points.append(_point((cx, cy, cz)))
    sources.camera_source = [Int64(data=0)] * full_xyz.shape[0]

    cloud_indexed = CloudIndexed()
    cloud_indexed.cloud_sources = sources
    cloud_indexed.indices = [Int64(data=int(i)) for i in obj_indices]

    mins, maxs = _pcd_bounds(env_xyz if env_xyz.size else full_xyz)
    workspace = _pad_workspace(mins, maxs, workspace_margin)

    approach_dir = decide_approach_direction(obj_xyz, StretchApproachConfig(max_lift_z=stretch_max_lift_z))

    gp = GraspParams()
    gp.approach_direction = _vector3(_normalize(approach_dir))
    flat_cams: List[float] = []
    for (cx, cy, cz) in cam_positions:
      flat_cams.extend([float(cx), float(cy), float(cz)])
    gp.camera_position = flat_cams
    tx, ty, tz, qx, qy, qz, qw = cam_to_base
    gp.transform_camera2base = _transform_from_rt((tx, ty, tz), (qx, qy, qz, qw))
    gp.workspace = workspace
    gp.enable_approach_dir_filtering = bool(enable_approach_filter)
    gp.approach_dir_threshold = float(approach_threshold_deg)

    req = DetectConstrainedGrasps.Request()
    req.cloud_indexed = cloud_indexed
    req.grasp_params = gp
    req.params_policy = params_policy

    future: Future = self.cli.call_async(req)
    rclpy.spin_until_future_complete(self, future, timeout_sec=120.0)
    if not future.done() or future.result() is None:
      raise RuntimeError("detect_constrained_grasps call failed or timed out")
    return future.result()

def visualize_input_pcds(pcd_obj=None, pcd_env=None):
    if not _HAS_O3D:
        print("Open3D not available. Skipping visualization.")
        return

    geometry_list = []

    if pcd_env is not None:
      print("Visualizing pcd_obj and pcd_env with Open3D...")
      pcd_env_o3d = o3d.geometry.PointCloud()
      pcd_env_o3d.points = o3d.utility.Vector3dVector(_to_numpy_xyz(pcd_env))
      pcd_env_o3d.paint_uniform_color([0.5, 0.5, 0.5]) # Gray for environment
      geometry_list.append(pcd_env_o3d)

    if pcd_obj is not None:
      pcd_obj_o3d = o3d.geometry.PointCloud()
      pcd_obj_o3d.points = o3d.utility.Vector3dVector(_to_numpy_xyz(pcd_obj))
      pcd_obj_o3d.paint_uniform_color([1.0, 0.0, 0.0]) # Red for object
      geometry_list.append(pcd_obj_o3d)
    
    # Add a coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    
    geometry_list.append(coord_frame)
    o3d.visualization.draw_geometries(geometry_list, 
                                      window_name="GPD Client Input", 
                                      width=1024, height=768)

def visualize_grasps(pcd_obj=None, pcd_env=None, grasps=[], max_grasps=10):
    if not _HAS_O3D:
        print("Open3D not available. Skipping grasp visualization.")
        return
    
    geoms = []
    
    # Environment point cloud (Gray)
    if pcd_env is not None:
      pcd_env_o3d = o3d.geometry.PointCloud()
      pcd_env_o3d.points = o3d.utility.Vector3dVector(_to_numpy_xyz(pcd_env))
      pcd_env_o3d.paint_uniform_color([0.5, 0.5, 0.5])
      geoms.append(pcd_env_o3d)
    
    # Object point cloud (Red)
    if pcd_obj is not None:
      pcd_obj_o3d = o3d.geometry.PointCloud()
      pcd_obj_o3d.points = o3d.utility.Vector3dVector(_to_numpy_xyz(pcd_obj))
      pcd_obj_o3d.paint_uniform_color([1.0, 0.0, 0.0])
      geoms.append(pcd_obj_o3d)
    
    # Coordinate system at origin
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    
    # Gripper dimensions (matching GPD defaults)
    hand_depth = 0.06
    hand_outer_diameter = 0.12
    finger_width = 0.01
    hand_height = 0.02
    
    print(f"Adding {min(len(grasps), max_grasps)} grasps to the scene...")
    for i, g in enumerate(grasps[:max_grasps]):
        # Construct 4x4 matrix from GPD GraspConfig
        T = np.eye(4)
        T[0:3, 0] = [g.approach.x, g.approach.y, g.approach.z]
        T[0:3, 1] = [g.binormal.x, g.binormal.y, g.binormal.z]
        T[0:3, 2] = [g.axis.x, g.axis.y, g.axis.z]
        T[0:3, 3] = [g.position.x, g.position.y, g.position.z]
        
        # Palm
        palm = o3d.geometry.TriangleMesh.create_box(width=hand_depth, height=hand_outer_diameter, depth=hand_height)
        palm.translate([-hand_depth, -hand_outer_diameter/2, -hand_height/2])
        
        # Use detected width for finger spacing
        w = g.width.data
        
        # Finger 1
        f1 = o3d.geometry.TriangleMesh.create_box(width=hand_depth, height=finger_width, depth=hand_height)
        f1.translate([0, w/2, -hand_height/2])
        
        # Finger 2
        f2 = o3d.geometry.TriangleMesh.create_box(width=hand_depth, height=finger_width, depth=hand_height)
        f2.translate([0, -w/2 - finger_width, -hand_height/2])
        
        gripper = palm + f1 + f2
        gripper.transform(T)
        
        # Color based on score (Green for positive, Red for negative)
        color = [0.1, 0.8, 0.1] if g.score.data > 0 else [0.8, 0.1, 0.1]
        gripper.paint_uniform_color(color)
        geoms.append(gripper)
        
    o3d.visualization.draw_geometries(geoms, window_name="GPD Grasps Visualization", 
                                      width=1024, height=768)

def main():
  rclpy.init()
  node = GpdConstrainedClient()
  # Example synthetic clouds
  env = np.random.uniform([-0.3,-0.3,0.65],[0.3,0.3,0.8], size=(5000,3)).astype(np.float32)
  obj = np.random.uniform([-0.05,-0.05,0.8],[0.05,0.05,0.85], size=(800,3)).astype(np.float32)

  visualize_input_pcds(obj, env)

  frame_id = "base_link"
  cam_positions = [(0.5, 0.0, 1.2)]
  cam_to_base = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
  try:
    res = node.call_with_pcds(
      pcd_obj=obj, pcd_env=env, frame_id=frame_id,
      cam_positions=cam_positions, cam_to_base=cam_to_base,
      params_policy=DetectConstrainedGrasps.Request.USE_CFG_FILE,
      approach_threshold_deg=25.0, workspace_margin=0.01,
      enable_approach_filter=True, stretch_max_lift_z=1.1,
    )
    node.get_logger().info(f"Got {len(res.grasp_configs.grasps)} grasps")
    
    if len(res.grasp_configs.grasps) > 0:
        visualize_grasps(obj, env, res.grasp_configs.grasps)
    else:
        print("No grasps to visualize.")

  except Exception as e:
    node.get_logger().error(f"Service call failed: {e}")
  finally:
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
  main()
