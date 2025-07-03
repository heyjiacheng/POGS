# RealSense 场景捕获使用说明

## 概述

`scene_capture.py` 是一个用于使用 RealSense 相机（D435 和 D405）和 UR5 机械臂捕获多视角场景数据的脚本。它会生成 NeRF 格式的数据集，包括 RGB 图像、深度图、相机位姿和点云。

## 前置条件

### 1. 硬件要求
- UR5 机械臂
- Intel RealSense D435 相机（安装在机械臂手腕）
- Intel RealSense D405 相机（固定在场景中作为第三视角）

### 2. 软件依赖
确保安装了以下 Python 包：
```bash
pip install pyrealsense2 open3d numpy opencv-python PyYAML tqdm
```

### 3. 必需的标定文件
运行场景捕获前，必须先运行相机标定：
```bash
python src/pogs/scripts/calibrate_cameras.py
```

标定会生成以下文件：
- `src/pogs/calibration_outputs/wrist_to_d435.tf` - 手腕到D435相机的变换
- `src/pogs/calibration_outputs/world_to_d405.tf` - 世界坐标系到D405相机的变换
- `src/pogs/calibration_outputs/realsense_calibration_trajectory.npy` - 机械臂轨迹

### 4. 相机配置文件
确保 `src/pogs/configs/camera_config.yaml` 包含正确的相机序列号：
```yaml
static_d405:
  exposure: 100
  flip_mode: false
  fps: 30
  gain: 31
  id: "130322272869"  # 替换为你的D405序列号
  resolution: 1080p
wrist_d435:
  exposure: 67
  flip_mode: true
  fps: 30
  gain: 28
  id: "819612070593"  # 替换为你的D435序列号
  resolution: 720p
```

## 使用方法

### 基本使用
```bash
python src/pogs/scripts/scene_capture.py --scene my_scene_name
```

### 参数说明
- `--scene`: （必需）场景名称，用于保存数据的文件夹名

## 输出文件结构

脚本会在 `src/pogs/data/utils/datasets/[scene_name]/` 下创建以下文件结构：

```
my_scene_name/
├── rgb/                    # RGB 图像
│   ├── frame_00001.png    # D405 静态相机图像
│   ├── frame_00002.png    # D435 第一个轨迹点图像
│   └── ...
├── depth/                  # 深度图（.npy 格式）
│   ├── frame_00001.npy
│   ├── frame_00002.npy
│   └── ...
├── poses/                  # 相机位姿矩阵
│   ├── 000.txt            # NeRF 格式的4x4变换矩阵
│   ├── 001.txt
│   └── ...
├── transforms.json         # NeRF 格式的数据集描述文件
├── sparse_pc.ply          # 合并的点云文件
└── camera_intrinsics.intr # 相机内参文件
```

## 工作流程

1. **初始化**: 加载配置文件和标定数据
2. **机械臂设置**: 移动机械臂到起始位置
3. **相机初始化**: 启动两台 RealSense 相机
4. **静态捕获**: 从固定的 D405 相机捕获第一帧
5. **轨迹执行**: 沿着预设轨迹移动机械臂，从手腕 D435 捕获多帧
6. **数据处理**: 合并点云，生成 NeRF 格式文件
7. **保存结果**: 保存所有数据到指定目录

## 故障排除

### 常见错误及解决方案

1. **标定文件缺失**
   ```
   ✗ Error loading calibration files: [Errno 2] No such file or directory
   ```
   **解决**: 先运行 `calibrate_cameras.py` 生成标定文件

2. **相机连接失败**
   ```
   ✗ Error initializing cameras: No device connected
   ```
   **解决**: 
   - 检查相机 USB 连接
   - 确认相机序列号正确
   - 尝试 `rs-enumerate-devices` 查看可用设备

3. **轨迹文件缺失**
   ```
   ✗ Trajectory file not found
   ```
   **解决**: 运行标定脚本生成轨迹文件

4. **机械臂连接失败**
   **解决**: 
   - 检查机械臂网络连接
   - 确认机械臂已启动并处于远程模式

### 检查相机序列号
使用以下命令查看连接的 RealSense 设备：
```bash
rs-enumerate-devices
```

## 数据格式说明

### transforms.json 格式
```json
{
  "frames": [
    {
      "file_path": "rgb/frame_00001.png",
      "depth_file_path": "depth/frame_00001.npy",
      "transform_matrix": [...],  // 4x4 相机位姿矩阵
      "fl_x": 640.0,             // 焦距 x
      "fl_y": 640.0,             // 焦距 y
      "cx": 320.0,               // 主点 x
      "cy": 240.0,               // 主点 y
      "w": 640,                  // 图像宽度
      "h": 480,                  // 图像高度
      "k1": 0.0,                 // 径向畸变系数
      "k2": 0.0,
      "k3": 0.0,
      "k4": 0.0,
      "p1": 0.0,                 // 切向畸变系数
      "p2": 0.0
    }
  ],
  "ply_file_path": "sparse_pc.ply"
}
```

### 深度图格式
- 格式：NumPy `.npy` 文件
- 单位：米（已从毫米转换）
- 数据类型：`float64`

## 与 NeRF 集成

生成的数据集可以直接用于 NeRF 训练：
```bash
# 使用 NeRFStudio
ns-train nerfacto --data path/to/my_scene_name

# 或使用其他 NeRF 实现
python train.py --datadir path/to/my_scene_name --dataset_type realsense
```

## 性能优化建议

1. **点云采样**: 脚本自动将点云限制在 100,000 点以下以提高性能
2. **相机预热**: 相机启动时会跳过前10帧以确保稳定性
3. **运动稳定**: 每个轨迹点之间有1秒等待时间确保稳定

## 高级配置

### 修改相机参数
在 `camera_config.yaml` 中调整：
- `fps`: 帧率（建议30）
- `exposure`: 曝光时间
- `gain`: 增益设置

### 修改轨迹
重新运行 `calibrate_cameras.py` 的教学模式来生成新轨迹

### 自定义分辨率
在 `scene_capture.py` 的 `RealSenseCamera` 初始化中修改 width/height 参数 