import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from ur5py.ur5 import UR5Robot
from autolab_core import RigidTransform
import time
import signal
import sys


class CyclicSubgoalExecutor:
    """循环执行JSON文件中定义的subgoals的控制器"""
    
    def __init__(self):
        """初始化机械臂环境"""
        self.robot = UR5Robot(gripper=1)
        self.clear_tcp()
        # 定义home position
        self.home_joints = np.array([0.07497743517160416, -2.0328524748431605, 1.277921199798584, -0.8172596136676233, -1.5602710882769983, 3.2171106338500977])
        self.stop_requested = False
        
        # 设置信号处理器以优雅地停止程序
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, sig, frame):
        """处理Ctrl+C信号"""
        print("\n\n收到停止信号，正在安全停止...")
        self.stop_requested = True
        
    def load_subgoals(self, json_file_path):
        """加载subgoals JSON文件"""
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        return data
    
    def apply_gripper_offset(self, pose):
        """为末端执行器姿态添加夹爪offset"""
        position = np.array(pose[:3])
        quat = np.array(pose[3:7])  # [qx, qy, qz, qw]
        
        # 创建旋转矩阵并应用offset
        rotation_matrix = R.from_quat(quat).as_matrix()
        z_offset = np.array([0, 0, 0.168])  # 16.8cm沿z轴offset
        z_offset_world = rotation_matrix @ z_offset
        
        # 应用offset
        new_position = position - z_offset_world
        
        return np.concatenate([new_position, quat])
    
    def process_pose_for_execution(self, pose):
        """将pose转换为机器人执行格式"""
        # pose格式: [x, y, z, qx, qy, qz, qw]
        position = pose[:3]
        quaternion = pose[3:7]
        
        # 使用ur5py的RigidTransform格式
        pose = RigidTransform(
            rotation=R.from_quat(quaternion).as_matrix(),
            translation=position,
            from_frame='tool',
            to_frame='world'
        )
        
        return pose
    
    def clear_tcp(self):
        """清除TCP设置"""
        tcp = RigidTransform(translation=np.array([0, 0, 0]), from_frame='tool', to_frame='wrist')
        self.robot.set_tcp(tcp)
    
    def move_to_home(self):
        """移动到home position"""
        print("移动到home position...")
        self.robot.move_joint(self.home_joints, vel=1.0, acc=0.1)
        self.robot.gripper.open()
        time.sleep(2.0)
        print("已到达home position")
    
    def execute_forward_cycle(self, subgoals):
        """正向执行subgoals（最后一步不释放）"""
        print("\n=== 开始正向执行 ===")
        
        for subgoal in sorted(subgoals, key=lambda x: x['stage']):
            if self.stop_requested:
                return False
                
            stage = subgoal['stage']
            subgoal_pose = subgoal['subgoal_pose']
            is_grasp_stage = subgoal['is_grasp_stage']
            is_release_stage = subgoal['is_release_stage']
            
            print(f"\n--- 正向执行Stage {stage} ---")
            
            # 添加夹爪offset
            pose_with_offset = self.apply_gripper_offset(subgoal_pose)
            processed_action = self.process_pose_for_execution(pose_with_offset)
            
            # 如果是抓取阶段，先打开夹爪
            if is_grasp_stage:
                print("打开夹爪准备抓取...")
                self.robot.gripper.open()
                time.sleep(1.0)
            
            # 执行移动到目标位置
            print("移动到目标位置...")
            self.robot.move_pose(processed_action, vel=0.5, acc=0.1)
            time.sleep(1.0)
            
            # 根据阶段类型控制夹爪
            if is_grasp_stage:
                print("执行抓取动作...")
                self.robot.gripper.close()
                time.sleep(1.0)
            elif is_release_stage:
                # 最后一步不释放，跳过释放动作
                print("跳过释放动作（保持抓取状态）...")
                pass
            
            print(f"正向Stage {stage} 完成")
            time.sleep(0.5)
        
        print("正向执行完成（保持抓取状态）")
        return True
    
    def execute_reverse_cycle(self, subgoals):
        """反向执行subgoals（在grasp stage释放物体）"""
        print("\n=== 开始反向执行 ===")
        
        # 反向排序subgoals
        for subgoal in sorted(subgoals, key=lambda x: x['stage'], reverse=True):
            if self.stop_requested:
                return False
                
            stage = subgoal['stage']
            subgoal_pose = subgoal['subgoal_pose']
            is_grasp_stage = subgoal['is_grasp_stage']
            is_release_stage = subgoal['is_release_stage']
            
            print(f"\n--- 反向执行Stage {stage} ---")
            
            # 添加夹爪offset
            pose_with_offset = self.apply_gripper_offset(subgoal_pose)
            processed_action = self.process_pose_for_execution(pose_with_offset)
            
            # 执行移动到目标位置
            print("移动到目标位置...")
            self.robot.move_pose(processed_action, vel=0.5, acc=0.1)
            time.sleep(1.0)
            
            # 在grasp stage释放物体
            if is_grasp_stage:
                print("在抓取位置释放物体...")
                self.robot.gripper.open()
                time.sleep(1.0)
            
            print(f"反向Stage {stage} 完成")
            time.sleep(0.5)
        
        print("反向执行完成")
        return True
    
    def execute_cyclic_subgoals(self, json_file_path):
        """循环执行subgoals"""
        # 加载subgoals数据
        data = self.load_subgoals(json_file_path)
        subgoals = data['subgoals']
        
        print(f"开始循环执行 {data['num_stages']} 个阶段的subgoals")
        print("按Ctrl+C停止程序")
        
        cycle_count = 0
        
        while not self.stop_requested:
            cycle_count += 1
            print(f"\n{'='*50}")
            print(f"开始第 {cycle_count} 个循环")
            print(f"{'='*50}")
            
            # 1. 正向执行（最后不释放）
            if not self.execute_forward_cycle(subgoals):
                break
            
            if self.stop_requested:
                break
                
            # 2. 反向执行（在grasp stage释放）
            if not self.execute_reverse_cycle(subgoals):
                break
            
            if self.stop_requested:
                break
                
            # 3. 回到home position
            print("\n--- 返回home position ---")
            self.move_to_home()
            
            if self.stop_requested:
                break
            
            print(f"第 {cycle_count} 个循环完成")
            time.sleep(1.0)  # 循环间短暂停顿
        
        print(f"\n程序停止，共完成 {cycle_count} 个循环")
        
        # 最终返回home position
        print("最终返回home position...")
        self.move_to_home()


def main():
    """主函数"""
    json_file_path = "/home/jiachengxu/workspace/master_thesis/POGS/outputs/action_subgoals.json"
    
    # 创建执行器
    executor = CyclicSubgoalExecutor()
    
    try:
        # 首先移动到home position
        executor.move_to_home()
        
        # 开始循环执行subgoals
        executor.execute_cyclic_subgoals(json_file_path)
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
    finally:
        # 断开连接
        print("断开机器人连接...")
        executor.robot.ur_c.disconnect()


if __name__ == "__main__":
    main()