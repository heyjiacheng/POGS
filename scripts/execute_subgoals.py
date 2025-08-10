import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from ur5py.ur5 import UR5Robot
from autolab_core import RigidTransform
import time


class SubgoalExecutor:
    """执行JSON文件中定义的subgoals的控制器"""
    
    def __init__(self):
        """初始化机械臂环境"""
        self.robot = UR5Robot(gripper=1)
        self.clear_tcp()
        # 定义home position
        self.home_joints = np.array([0.07497743517160416, -2.0328524748431605, 1.277921199798584, -0.8172596136676233, -1.5602710882769983, 3.2171106338500977])
        
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
    
    def return_home(self):
        """任务完成后返回home position"""
        self.move_to_home()
    
    def execute_subgoals(self, json_file_path):
        """执行JSON文件中的所有subgoals"""
        # 加载subgoals数据
        data = self.load_subgoals(json_file_path)
        
        print(f"开始执行 {data['num_stages']} 个阶段的subgoals")
        
        # 按stage顺序执行
        for subgoal in sorted(data['subgoals'], key=lambda x: x['stage']):
            stage = subgoal['stage']
            subgoal_pose = subgoal['subgoal_pose']
            is_grasp_stage = subgoal['is_grasp_stage']
            is_release_stage = subgoal['is_release_stage']
            
            print(f"\n=== 执行Stage {stage} ===")
            print(f"目标姿态: {subgoal_pose}")
            print(f"抓取阶段: {is_grasp_stage}")
            print(f"释放阶段: {is_release_stage}")
            
            # 添加夹爪offset
            pose_with_offset = self.apply_gripper_offset(subgoal_pose)
            print(f"添加offset后的姿态: {pose_with_offset}")
            
            # 转换为机器人执行格式
            processed_action = self.process_pose_for_execution(pose_with_offset)
            print(f"处理后的动作: {processed_action}")
            
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
                print("执行释放动作...")
                self.robot.gripper.open()
                time.sleep(1.0)
            
            print(f"Stage {stage} 完成")
            time.sleep(0.5)  # 短暂停顿
        
        print("\n所有subgoals执行完成!")
        
        # 返回home position
        print("返回home position...")
        self.return_home()


def main():
    """主函数"""
    json_file_path = "/home/jiachengxu/workspace/master_thesis/POGS/outputs/action_subgoals.json"
    
    # 创建执行器并执行subgoals
    executor = SubgoalExecutor()
    
    # 首先移动到home position
    executor.move_to_home()
    
    # 执行subgoals
    executor.execute_subgoals(json_file_path)
    
    # 断开连接
    executor.robot.ur_c.disconnect()


if __name__ == "__main__":
    main()