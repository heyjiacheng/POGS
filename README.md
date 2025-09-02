## Persistent Object Gaussian Splat (POGS) for Tracking Human and Robot Manipulation of Irregularly Shaped Objects

#### Download Models and Data
##### Model
Download trained models from [here](https://drive.google.com/drive/folders/1tBHKf60K8DLM5arm-Chyf7jxkzOr5zGl?usp=sharing) and copy them into the `checkpoints/` folder.
##### Test data
Download the test data from [here](https://drive.google.com/drive/folders/1TqpM2wHAAo0j3i1neu3Xeru3_WnsYQnx?usp=sharing) and copy them them into the `test_data/` folder.

## Usage
```
pixi shell
```
### Calibrate wrist mounted and third person cameras
Before training/tracking POGS, make sure wrist mounted camera and third-person view camera are calibrated. We use an Aruco marker for the calibration
```
cd src/pogs/scripts
python calibrate_cameras.py
```
### Rekep
Run Rekep and get path
```
pixi r rekep
```

### Scene Capture
Script used to perform hemisphere capture with robot on tabletop scene. We used manual trajectory but you can also put the robot in "teach" mode to capture trajectory.
```
python src/pogs/scripts/scene_capture.py --scene box
```

### Train POGS
Script used to train the POGS for 3000 steps
```
ns-train pogs --data /home/jiachengxu/workspace/master_thesis/POGS/src/pogs/scripts/../data/utils/datasets/box
```
Once the POGS has completed training, there are N steps to then actually define/save the object clusters.
1. Hit the cluster scene button.
2. It will take 10-20 seconds, but then after, you should see your objects as specific clusters. If not, hit Toggle RGB/Cluster and try to cluster the scene again but change the Cluster Eps (lower normally works better).
3. Once you have your scene clustered, hit Toggle RGB/Cluster.
4. Then, hit Click and click on your desired object (green ball will appear on object).
5. Hit Crop to Click, and it should isolate the object.
6. A draggable coordinate frame will pop up to indicate the object's origin, drag it to where you want it to be. (For experiments, this was what we used to align for object reset or tool servoing)
7. Hit Add Crop to Group List
8. Repeat steps 4-7 for all objects in scene
9. Hit View Crop Group List
Once you have trained the POGS, make sure you have the config file and checkpoint directory from the terminal saved.

### View Gsplat
```
ns-viewer --load-config /home/jiachengxu/workspace/master_thesis/POGS/outputs/box/pogs/2025-09-02_204315/config.yml --viewer.websocket-port 8007
```

### Output filtered end pose
```
python scripts/sample_and_filter_poses.py
```

### Run Bi-RRT to get path
```
python scripts/BiRRT_Cons.py
```

### Visualize animation that equal to action
```
python scripts/generate_action_animation.py
```

### Execute on UR5
```
python scripts/execute_subgoals.py
```

### Cycle execute UR5
```
python scripts/cyclic_execute_subgoals.py
```

## Bibtex
If you find POGS useful for your work please consider citing:
```
@article{yu2025pogs,
  author    = {Yu, Justin and Hari, Kush and El-Refai, Karim and Dalil, Arnav and Kerr, Justin and Kim, Chung-Min and Cheng, Richard, and Irshad, Muhammad Z. and Goldberg, Ken},
  title     = {Persistent Object Gaussian Splat (POGS) for Tracking Human and Robot Manipulation of Irregularly Shaped Objects},
  journal   = {ICRA},
  year      = {2025},
}
```
