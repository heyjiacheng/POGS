from __future__ import annotations

from typing import List, Tuple, Type
import random

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json
from pathlib import Path
import open3d as o3d
import warnings

# 抑制matplotlib警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
plt.rcParams.update({'figure.max_open_warning': 0})
# ---------------------------------------------------------------------------
#  Geometry helpers (unchanged except for inflation util)
# ---------------------------------------------------------------------------
Vec3   = np.ndarray
Bounds = List[Tuple[float, float]]

class Box:
    def __init__(self, min_corner, max_corner):
        self.min = np.minimum(min_corner, max_corner).astype(float)
        self.max = np.maximum(min_corner, max_corner).astype(float)

    def segment(self, p: Vec3, q: Vec3) -> bool:
        """Returns *True* if the segment 𝑝→𝑞 collides with the box."""
        d = q - p
        tmin, tmax = 0.0, 1.0
        for i in range(3):
            if abs(d[i]) < 1e-12:
                if p[i] < self.min[i] or p[i] > self.max[i]:
                    return True
            else:
                t1 = (self.min[i] - p[i]) / d[i]
                t2 = (self.max[i] - p[i]) / d[i]
                t_enter, t_exit = min(t1, t2), max(t1, t2)
                tmin = max(tmin, t_enter)
                tmax = min(tmax, t_exit)
                if tmin > tmax:
                    return False
        return True  # Any remaining overlap → collision

class OpenBox(Box):
    """A box that is *open* at the +Z face (see original comments)."""
    def segment(self, p: Vec3, q: Vec3) -> bool:
        # 1) Both endpoints inside → free.
        inside_p = np.all(p >= self.min) and np.all(p <= self.max)
        inside_q = np.all(q >= self.min) and np.all(q <= self.max)
        if inside_p and inside_q:
            return False
        # 2) Crossing the top is also okay.
        d = q - p
        if abs(d[2]) > 1e-12:
            t_top = (self.max[2] - p[2]) / d[2]
            if 0.0 <= t_top <= 1.0:
                x_top = p[0] + t_top * d[0]
                y_top = p[1] + t_top * d[1]
                if (self.min[0] <= x_top <= self.max[0] and
                    self.min[1] <= y_top <= self.max[1]):
                    return False
        return super().segment(p, q)

# ---------------------------------------------------------------------------
#  Visualisation utility (unchanged)
# ---------------------------------------------------------------------------

def draw_box(ax, box, alpha=0.05, facecolor='black', edgecolor='k', linewidth=1):
    x0, y0, z0 = box.min
    x1, y1, z1 = box.max
    verts = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
    ])
    faces = [
        [verts[0], verts[1], verts[2], verts[3]],
        [verts[0], verts[1], verts[5], verts[4]],
        [verts[1], verts[2], verts[6], verts[5]],
        [verts[2], verts[3], verts[7], verts[6]],
        [verts[3], verts[0], verts[4], verts[7]],
    ]
    ax.add_collection3d(Poly3DCollection(faces, alpha=alpha,
                                         facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth))

# ---------------------------------------------------------------------------
#  Node + BiRRT* implementation with rigid‑body (box) support
# ---------------------------------------------------------------------------

class Node:  # cost‑aware version
    def __init__(self, pos: Vec3, parent: "Node|None" = None, cost: float = 0.0):
        self.pos: Vec3 = np.asarray(pos, dtype=float)
        self.parent: Node | None = parent
        self.cost: float = cost  # path cost from *its* root

class BiRRTStar:
    """Bidirectional, cost‑aware (RRT*) planner ***for a point robot***."""

    def __init__(self,
                 start: Vec3,
                 goal: Vec3,
                 bounds: Bounds,
                 step: float = 0.2,
                 max_iter: int = 5000,
                 goal_rate: float = 0.05,
                 obstacles: List[Box] | None = None,
                 ax=None):
        self.bounds      = bounds
        self.step        = step
        self.max_iter    = max_iter
        self.goal_rate   = goal_rate
        self.obstacles   = obstacles or []
        self.ax          = ax

        # Two trees — each keeps a list[Node]
        self.start_root  = Node(np.array(start))
        self.goal_root   = Node(np.array(goal))
        self.trees       = [ [self.start_root], [self.goal_root] ]  # 0 = start, 1 = goal

        # Best current solution
        self.best_cost   = float('inf')
        self.best_pair: Tuple[Node, Node] | None = None

        # Rewire radius (γ * (log n / n)^(1/3)), simple constant‑factor approximation
        self.gamma       = 2.0  # Tunable

    # -------------------------------------------------------------------
    #  Core helpers
    # -------------------------------------------------------------------
    def sample(self) -> Vec3:
        if random.random() < self.goal_rate:
            return self.goal_root.pos  # Bias towards *exact* goal
        return np.array([random.uniform(lo, hi) for lo, hi in self.bounds])

    def nearest(self, tree: List[Node], p: Vec3) -> Node:
        return min(tree, key=lambda n: np.linalg.norm(n.pos - p))

    def steer(self, from_node: Node, to_pos: Vec3) -> Vec3:
        d = to_pos - from_node.pos
        l = np.linalg.norm(d)
        if l <= self.step:
            return to_pos
        return from_node.pos + (self.step / l) * d

    def collision_free(self, p: Vec3, q: Vec3) -> bool:
        return all(not obs.segment(p, q) for obs in self.obstacles)

    def near_nodes(self, tree: List[Node], p: Vec3) -> List[Node]:
        # Radius ≈ γ (log n / n) ^ 1/3   (works for 3‑D)
        n = len(tree) + 1
        r = min(self.gamma * (np.log(n) / n) ** (1/3), self.step * 10)
        return [node for node in tree if np.linalg.norm(node.pos - p) <= r]

    # -------------------------------------------------------------------
    #  Main planning loop (unchanged)
    # -------------------------------------------------------------------
    def plan(self):
        for it in range(self.max_iter):
            rand_p = self.sample()

            # Alternate between the two trees — ensures balanced growth
            active_idx       = it % 2
            other_idx        = 1 - active_idx
            active_tree      = self.trees[active_idx]
            other_tree       = self.trees[other_idx]

            nearest          = self.nearest(active_tree, rand_p)
            new_pos          = self.steer(nearest, rand_p)
            if not self.collision_free(nearest.pos, new_pos):
                continue

            # Choose parent with minimal cost (RRT*)
            near_nodes       = self.near_nodes(active_tree, new_pos)
            parent           = nearest
            min_cost         = nearest.cost + np.linalg.norm(nearest.pos - new_pos)
            for node in near_nodes:
                c = node.cost + np.linalg.norm(node.pos - new_pos)
                if c < min_cost and self.collision_free(node.pos, new_pos):
                    parent, min_cost = node, c

            new_node = Node(new_pos, parent, min_cost)
            active_tree.append(new_node)

            # Rewire nearby nodes towards *new_node*
            for node in near_nodes:
                c_through_new = new_node.cost + np.linalg.norm(node.pos - new_node.pos)
                if c_through_new < node.cost and self.collision_free(node.pos, new_node.pos):
                    node.parent = new_node
                    node.cost   = c_through_new

            # Visualise this edge (optional)
            if self.ax is not None:
                p, q = parent.pos, new_node.pos
                color = 'red' if active_idx == 0 else 'purple'
                self.ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]],
                             color=color, linewidth=0.5, alpha=0.3, zorder=1)
                # plt.pause(0.0001)

            # Attempt to connect the newly added node towards the *other* tree
            other_near  = self.nearest(other_tree, new_node.pos)
            if np.linalg.norm(other_near.pos - new_node.pos) <= self.step and \
               self.collision_free(other_near.pos, new_node.pos):
                total_cost = new_node.cost + other_near.cost + \
                             np.linalg.norm(other_near.pos - new_node.pos)
                if total_cost < self.best_cost:
                    self.best_cost = total_cost
                    self.best_pair = (new_node, other_near)

            # Early exit if we already found a pretty good path
            if self.best_cost < float('inf') and it > 0 and \
               it % 50 == 0:  # allow some improvement iterations
                break

        return self._build_path()

    # -------------------------------------------------------------------
    #  Path recovery
    # -------------------------------------------------------------------
    def _trace(self, node: Node) -> List[Vec3]:
        path = []
        while node is not None:
            path.append(node.pos)
            node = node.parent
        return path

    def _build_path(self) -> List[Vec3] | None:
        if self.best_pair is None:
            return None
        a, b = self.best_pair
        path_start = self._trace(a)
        path_goal  = self._trace(b)
        return path_start[::-1] + path_goal  # from start → goal

# ---------------------------------------------------------------------------
#  Extended planner for a rectangular rigid body (axis‑aligned, no rotation)
# ---------------------------------------------------------------------------

class BiRRTStarRigid(BiRRTStar):
    """BiRRT* that plans for an *axis‑aligned rectangular box* robot.

    The robot is assumed to keep a fixed orientation (no roll/pitch/yaw), so its
    configuration is fully described by the 3‑D coordinates of its centre.
    """

    def __init__(self,
                 start: Vec3,
                 goal: Vec3,
                 bounds: Bounds,
                 robot_size: Tuple[float, float, float],  # (dx, dy, dz)
                 step: float = 0.2,
                 max_iter: int = 5000,
                 goal_rate: float = 0.05,
                 obstacles: List[Box] | None = None,
                 ax=None):
        self.robot_half = np.asarray(robot_size, dtype=float) / 2.0

        # --- Shrink sampling bounds so that the *entire* box stays inside scene
        shrunk_bounds: Bounds = [ (lo + self.robot_half[i], hi - self.robot_half[i])
                                  for i, (lo, hi) in enumerate(bounds) ]

        # --- Inflate each obstacle by Minkowski sum with the robot box
        inflated_obs: List[Box] = []
        if obstacles is not None:
            for obs in obstacles:
                obs_cls: Type[Box] = obs.__class__  # Box or OpenBox
                new_min = obs.min - self.robot_half
                new_max = obs.max + self.robot_half
                inflated_obs.append(obs_cls(new_min, new_max))

        super().__init__(start, goal, shrunk_bounds, step, max_iter,
                         goal_rate, inflated_obs, ax)

    # No further changes are needed because the parent class always works in the
    # *shrunk* free space where the robot is treated as a point.

# ---------------------------------------------------------------------------
#  Demo / unit test
# ---------------------------------------------------------------------------

def draw_robot(ax, centre: Vec3, size, **kwargs):
    """绘制机器人的3D表示"""
    half = np.asarray(size) / 2.0
    box = Box(centre - half, centre + half)
    draw_box(ax, box, **kwargs)

def cluster_static_objects(pcd_static, eps=0.05, min_points=50):
    """对静态物体点云进行聚类，分离不同的物体"""
    points = np.asarray(pcd_static.points)
    print(f"开始聚类静态物体，总点数: {len(points)}")
    
    # 使用DBSCAN聚类
    labels = np.array(pcd_static.cluster_dbscan(eps=eps, min_points=min_points))
    max_label = labels.max()
    print(f"聚类完成，发现 {max_label + 1} 个物体（不包括噪声）")
    
    clusters = []
    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster_points = points[cluster_indices]
        
        if len(cluster_points) > min_points:  # 过滤太小的聚类
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
            clusters.append(cluster_pcd)
            print(f"物体 {i}: {len(cluster_points)} 个点")
    
    return clusters

def load_scene_data():
    """加载场景数据：起点、终点、机器人尺寸和障碍物"""
    # 读取subgoals.json文件,获取start和goal的xyz坐标
    subgoals_path = Path("outputs/new_subgoals.json")
    with open(subgoals_path, "r") as f:
        subgoals = json.load(f)
    
    start_pose = subgoals["subgoals"][0]["subgoal_pose"]
    start_xyz = np.array(start_pose[:3])
    goal_pose = subgoals["subgoals"][-1]["subgoal_pose"]
    goal_xyz = np.array(goal_pose[:3])
    
    print(f"起始点: {start_xyz.tolist()}")
    print(f"目标点: {goal_xyz.tolist()}")

    # 读目标物的点云信息获得最小包围盒
    pcd_moving = o3d.io.read_point_cloud("outputs/scene_animation/subgoal_1/moving_object.ply")
    moving_aabb = pcd_moving.get_axis_aligned_bounding_box()
    robot_size = tuple(moving_aabb.get_max_bound() - moving_aabb.get_min_bound())
    moving_center = moving_aabb.get_center()
    print(f"目标物体尺寸: {robot_size}")
    print(f"目标物体中心: {moving_center}")

    # 读取静态物体点云信息并进行聚类
    pcd_static = o3d.io.read_point_cloud("outputs/scene_animation/subgoal_1/static_objects.ply")
    
    # 对静态物体进行聚类
    static_clusters = cluster_static_objects(pcd_static, eps=0.03, min_points=100)
    
    obstacles = []
    moving_min, moving_max = moving_aabb.get_min_bound(), moving_aabb.get_max_bound()
    
    for i, cluster_pcd in enumerate(static_clusters):
        cluster_aabb = cluster_pcd.get_axis_aligned_bounding_box()
        cluster_min = cluster_aabb.get_min_bound()
        cluster_max = cluster_aabb.get_max_bound()
        cluster_center = cluster_aabb.get_center()
        
        # 检查此聚类是否与目标物体重叠
        overlap_x = max(0, min(moving_max[0], cluster_max[0]) - max(moving_min[0], cluster_min[0]))
        overlap_y = max(0, min(moving_max[1], cluster_max[1]) - max(moving_min[1], cluster_min[1]))
        overlap_z = max(0, min(moving_max[2], cluster_max[2]) - max(moving_min[2], cluster_min[2]))
        
        if overlap_x > 0.01 and overlap_y > 0.01 and overlap_z > 0.01:
            print(f"跳过物体 {i}（与目标物体重叠）: 中心={cluster_center}")
            continue
        
        # 添加为障碍物（使用OpenBox，假设都是容器类障碍物）
        obstacles.append(OpenBox(cluster_min, cluster_max))
        print(f"障碍物 {len(obstacles)}: 中心={cluster_center}, 尺寸={cluster_max - cluster_min}")
    
    print(f"总共创建了 {len(obstacles)} 个障碍物")
    
    return start_xyz, goal_xyz, robot_size, obstacles, subgoals

def setup_visualization(start_xyz, goal_xyz, robot_size, obstacles):
    """设置3D可视化环境"""
    robot_color = 'gold'
    plt.ion()
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 只绘制起始位置的目标物体
    draw_robot(ax, start_xyz, robot_size, alpha=0.5, facecolor='orange', edgecolor='red', linewidth=2)
    
    # 绘制障碍物（为不同障碍物使用淡色）
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink', 'lightgray']
    for i, box in enumerate(obstacles):
        color_idx = i % len(colors)
        draw_box(ax, box, alpha=0.1, facecolor=colors[color_idx], edgecolor='gray')
    
    return fig, ax

def plan_path(start_xyz, goal_xyz, robot_size, obstacles, ax=None):
    """执行路径规划"""
    bounds = [(-0.5, 0.5), (-0.5, 0.5), (0, 0.5)]
    
    print("开始BiRRT*路径规划...")
    planner = BiRRTStarRigid(
        start_xyz.tolist(), goal_xyz.tolist(), bounds, robot_size,
        step=0.03, max_iter=8000, goal_rate=0.05,
        obstacles=obstacles, ax=ax  # 传入ax以显示探索过程
    )
    
    path = planner.plan()
    if path is None:
        print('[失败] 未找到可行路径')
        return None, None
    
    path = np.array(path)
    print(f"路径规划成功! 路径长度: {len(path)}")
    
    # 确保路径方向正确
    if not np.allclose(path[0], start_xyz, atol=1e-3):
        path = path[::-1]
        print("路径方向已修正")
    
    return path, planner

def generate_output_json(path, subgoals, robot_size):
    """生成与原格式兼容的输出JSON"""
    first_quaternion = subgoals["subgoals"][0]["subgoal_pose"][3:7]
    last_quaternion = subgoals["subgoals"][-1]["subgoal_pose"][3:7]
    
    new_subgoals = []
    for i, point in enumerate(path):
        if i == len(path) - 1:
            quaternion = last_quaternion
            is_release_stage = True
            is_grasp_stage = False
        elif i == 0:
            quaternion = first_quaternion
            is_grasp_stage = True
            is_release_stage = False
        else:
            quaternion = first_quaternion
            is_grasp_stage = False
            is_release_stage = False
        
        subgoal_pose = point.tolist() + quaternion
        
        if i == 0:
            current_ee_pose = subgoal_pose.copy()
        else:
            prev_point = path[i-1]
            prev_quaternion = first_quaternion
            current_ee_pose = prev_point.tolist() + prev_quaternion
        
        subgoal = {
            "stage": i + 1,
            "subgoal_pose": subgoal_pose,
            "current_ee_pose": current_ee_pose,
            "is_grasp_stage": is_grasp_stage,
            "is_release_stage": is_release_stage
        }
        new_subgoals.append(subgoal)
    
    output_data = {
        "num_stages": len(path),
        "subgoals": new_subgoals,
        "metadata": {
            "modified": True,
            "original_file": "BiRRT*Cons.py generated path",
            "path_planning_algorithm": "BiRRT*",
            "robot_size": robot_size,
            "generation_timestamp": "2025-08-04"
        }
    }
    
    output_path = Path("outputs/action_subgoals.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print("路径数据已保存至 outputs/action_subgoals.json")

def visualize_final_path(ax, path, start_xyz, goal_xyz, robot_size, obstacles, show_exploration=True):
    """可视化最终路径，可选择是否显示探索过程"""
    bounds = [(-0.5, 0.5), (-0.5, 0.5), (0, 0.5)]
    
    if not show_exploration:
        # 如果不显示探索过程，清除之前的内容
        ax.clear()
        
        # 重新绘制环境
        draw_robot(ax, start_xyz, robot_size, alpha=0.6, facecolor='orange', edgecolor='red', linewidth=2)
        # 为不同障碍物使用不同颜色
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink', 'lightgray']
        edge_colors = ['navy', 'darkred', 'darkgreen', 'orange', 'purple', 'black']
        for i, box in enumerate(obstacles):
            color_idx = i % len(colors)
            draw_box(ax, box, alpha=0.3, 
                    facecolor=colors[color_idx], 
                    edgecolor=edge_colors[color_idx], 
                    linewidth=2)
    else:
        # 如果显示探索过程，只需要在现有图上添加最终路径和环境信息
        draw_robot(ax, start_xyz, robot_size, alpha=0.6, facecolor='orange', edgecolor='red', linewidth=2)
        # 为不同障碍物使用不同颜色
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink', 'lightgray']
        edge_colors = ['navy', 'darkred', 'darkgreen', 'orange', 'purple', 'black']
        for i, box in enumerate(obstacles):
            color_idx = i % len(colors)
            draw_box(ax, box, alpha=0.3, 
                    facecolor=colors[color_idx], 
                    edgecolor=edge_colors[color_idx], 
                    linewidth=2)
    
    # 绘制最终路径（更粗的线条，高层级显示）
    ax.plot(path[:,0], path[:,1], path[:,2], color='lime', linewidth=8,
            zorder=20, alpha=0.9)
    
    # 标记关键点（更大更醒目）
    ax.scatter(*path[0], marker='o', s=200, color='red', 
              zorder=25, edgecolor='darkred', linewidth=3)
    ax.scatter(*path[-1], marker='^', s=200, color='blue', 
              zorder=25, edgecolor='darkblue', linewidth=3)
    
    # 沿路径标记中间点（更清晰）
    for i in range(1, len(path) - 1):
        ax.scatter(*path[i], marker='o', s=80, color='yellow', alpha=0.9, zorder=22, edgecolor='darkgreen', linewidth=1)
    
    # 添加方向箭头（更明显的箭头）
    arrow_step = max(1, len(path) // 6)  # 适当的箭头密度
    for i in range(0, len(path) - 1, arrow_step):
        start_pt = path[i]
        end_pt = path[i + 1]
        direction = end_pt - start_pt
        ax.quiver(start_pt[0], start_pt[1], start_pt[2],
                 direction[0], direction[1], direction[2],
                 color='darkgreen', alpha=1.0, arrow_length_ratio=0.2, 
                 linewidth=3, zorder=21)
    
    # 设置图表样式
    ax.set_xlim(*bounds[0]); ax.set_ylim(*bounds[1]); ax.set_zlim(*bounds[2])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=45)
    
    # 移除所有注释
    ax.set_title('')
    ax.legend().set_visible(False) if ax.get_legend() else None
    # ax.grid(False)
    
    return ax.figure

def main():
    """主函数：执行完整的路径规划流程"""
    try:
        # 1. 加载场景数据
        start_xyz, goal_xyz, robot_size, obstacles, subgoals = load_scene_data()
        
        # 2. 设置可视化
        fig, ax = setup_visualization(start_xyz, goal_xyz, robot_size, obstacles)
        
        # 3. 执行路径规划（显示探索过程）
        path, planner = plan_path(start_xyz, goal_xyz, robot_size, obstacles, ax)
        if path is None:
            return
        
        # 4. 生成输出文件
        generate_output_json(path, subgoals, robot_size)
        
        # 5. 可视化最终结果（保留探索过程）
        fig = visualize_final_path(ax, path, start_xyz, goal_xyz, robot_size, obstacles, show_exploration=True)
        fig.tight_layout()

    
        # 6. 保存可视化图片
        plt.ioff()
        fig_dir = Path("outputs/figs")
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        # 静默保存图片
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.savefig(fig_dir / "path_visualization.jpg", dpi=300, bbox_inches='tight')
            
        print(f"路径可视化图片已保存至 {fig_dir / 'path_visualization.jpg'}")
        plt.show()
    
    except Exception as e:
        print(f"路径规划过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
