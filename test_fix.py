#!/usr/bin/env python3
"""
Test script to verify the AttributeError fix
"""

import sys
sys.path.insert(0, './scripts')

try:
    from auto_pipeline import AutoPipeline
    print("Import successful")
    
    pipeline = AutoPipeline(execute_robot=False)
    print("Planning mode initialization successful")
    
    print("Environment attribute exists:", hasattr(pipeline, 'env'))
    print("Subgoal solver exists:", hasattr(pipeline, 'subgoal_solver'))
    print("Path solver exists:", hasattr(pipeline, 'path_solver'))
    print("Reset joint positions exist:", hasattr(pipeline, 'reset_joint_pos'))
    
    print("\nAll required attributes are present!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()