#!/usr/bin/env python3
"""
Verification script to test the fixes for POGS viewer export functionality
"""

import sys
import os
from pathlib import Path

def check_fix():
    """Check if the fix is correctly applied"""
    
    pipeline_file = Path("src/pogs/pogs_pipeline.py")
    
    if not pipeline_file.exists():
        print("❌ Pipeline file not found")
        return False
    
    with open(pipeline_file, 'r') as f:
        content = f.read()
    
    # Check if the fix is present
    if "if len(self.state_stack) > 0:" in content:
        print("✅ Fix applied: state_stack length check added")
        
        # Check if both branches are present
        if "else:" in content and "No previous state, use current model state" in content:
            print("✅ Fix applied: fallback to current model state added")
            
            # Check if the other fix is present
            if "if len(self.crop_group_list) > 0 and len(self.state_stack) > 0:" in content:
                print("✅ Fix applied: crop group list safety check added")
                return True
            else:
                print("⚠️  Partial fix: crop group list safety check missing")
                return False
        else:
            print("⚠️  Partial fix: fallback branch missing")
            return False
    else:
        print("❌ Fix not applied: state_stack length check missing")
        return False

def main():
    print("🔍 Verifying POGS viewer export fix...")
    print()
    
    if check_fix():
        print()
        print("✅ All fixes successfully applied!")
        print()
        print("📋 What was fixed:")
        print("1. Added state_stack length check in _export_visible_gaussians")
        print("2. Added fallback to current model state when no previous state exists")
        print("3. Added safety check for crop group list operations")
        print()
        print("🎯 How to test:")
        print("1. Run: ns-viewer --load-config outputs/box/pogs/2025-07-18_132729/config.yml")
        print("2. Wait for the viewer to load")
        print("3. Click 'Export Visible Gaussians' in the UI")
        print("4. Check outputs/box/ for exported PLY files")
        print()
        print("📁 Expected output files:")
        print("- outputs/box/prime_seg_gaussians.ply")
        print("- outputs/box/prime_full_gaussians.ply")
        
    else:
        print()
        print("❌ Fix verification failed!")
        print("Please check the pipeline file and apply the fixes manually.")
        
    return 0 if check_fix() else 1

if __name__ == "__main__":
    sys.exit(main())