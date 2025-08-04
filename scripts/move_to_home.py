#!/usr/bin/env python3
"""
Move robot arm to home position
Simple script to initialize robot to safe home position
"""

import numpy as np
from ur5py.ur5 import UR5Robot
from autolab_core import RigidTransform
import time


def clear_tcp(robot):
    """Clear TCP settings"""
    tcp = RigidTransform(translation=np.array([0, 0, 0]), from_frame='tool', to_frame='wrist')
    robot.set_tcp(tcp)


def move_to_home():
    """Move robot to home position"""
    print("Initializing robot...")
    robot = UR5Robot(gripper=1)
    clear_tcp(robot)
    
    # Define home position (same as execute_subgoals.py)
    home_joints = np.array([0.07497743517160416, -2.0328524748431605, 1.277921199798584, -0.8172596136676233, -1.5602710882769983, 3.2171106338500977])
    
    print("Moving robot to home position...")
    robot.move_joint(home_joints, vel=1.0, acc=0.1)
    robot.gripper.open()
    time.sleep(2.0)
    print("Robot reached home position")
    
    # Disconnect
    robot.ur_c.disconnect()
    print("Robot disconnected")


if __name__ == "__main__":
    move_to_home()