# 智能导航系统评估数据集规范

## 1. 数据集目录结构

```
test_dataset/
├── blindpath/              # 盲道导航场景
│   ├── straight/           # 直行盲道
│   │   ├── indoor_01.mp4   # 室内直行
│   │   ├── outdoor_01.mp4  # 室外直行
│   │   └── ...
│   ├── curve/              # 转弯盲道
│   │   ├── left_01.mp4     # 左转
│   │   ├── right_01.mp4    # 右转
│   │   └── ...
│   └── broken/             # 盲道中断/损坏
│       ├── missing_01.mp4  # 盲道缺失
│       ├── blocked_01.mp4  # 盲道被阻挡
│       └── ...
├── crossing/               # 过马路场景
│   ├── zebra_crossing/     # 斑马线
│   │   ├── simple_01.mp4   # 简单斑马线
│   │   ├── complex_01.mp4  # 复杂斑马线
│   │   └── ...
│   ├── traffic_light/      # 红绿灯
│   │   ├── green_01.mp4    # 绿灯
│   │   ├── red_01.mp4      # 红灯
│   │   └── ...
│   └── pedestrian/         # 行人干扰
│       ├── crossing_01.mp4 # 行人穿越
│       └── ...
└── obstacles/             # 障碍物检测
    ├── static/             # 静态障碍
    │   ├── pole_01.mp4     # 电杆
    │   ├── box_01.mp4      # 箱子
    │   └── ...
    └── dynamic/            # 动态障碍
        ├── person_01.mp4   # 行人
        ├── car_01.mp4      # 车辆
        └── ...
```

## 2. 数据标注规范

### 2.1 视频级别标注

每个视频对应一个 JSON 标注文件：

```json
{
  "video_id": "blindpath_straight_indoor_01",
  "category": "blindpath",
  "scenario": "straight",
  "duration": 15.2,
  "fps": 30,
  "resolution": "640x480",
  "difficulty": "easy",
  "ground_truth": {
    "path_type": "straight",
    "blind_path_quality": "good",
    "main_challenges": [],
    "expected_guidance": ["保持直行"]
  }
}
```

### 2.2 帧级别标注（可选，用于细粒度评估）

```json
{
  "frame_00100": {
    "timestamp": 3.33,
    "blind_path_visible": true,
    "deviation_angle": 2.5,
    "deviation_offset": 0.05,
    "obstacles": [],
    "expected_action": "straight"
  },
  "frame_00150": {
    "timestamp": 5.00,
    "blind_path_visible": true,
    "deviation_angle": 8.0,
    "deviation_offset": 0.15,
    "obstacles": [],
    "expected_action": "turn_left"
  }
}
```

## 3. 数据收集指南

### 3.1 场景覆盖

| 场景类型 | 最少样本数 | 难度分布 |
|:---|:---:|:---|
| 简单场景（直行、无障碍） | 40% | easy |
| 中等场景（转弯、稀疏行人） | 35% | medium |
| 困难场景（盲道中断、密集行人） | 25% | hard |

### 3.2 录制规范

1. **设备要求**
   - 分辨率：640x480 或更高
   - 帧率：≥ 25 FPS
   - 稳定性：使用稳定器或三脚架

2. **环境要求**
   - 光照：晴天、阴天、室内均匀光照
   - 角度：平视（模拟盲人视角）
   - 高度：1.2-1.5 米（盲人手持设备高度）

3. **时长要求**
   - 每个视频：10-30 秒
   - 单个场景至少 3 次重复（不同光照/角度）

### 3.3 标注工具推荐

- **CVAT** (https://github.com/opencv/cvat)
- **LabelMe** (https://github.com/wkentaro/labelme)
- **VGG Image Annotator** (VIA)

## 4. 数据增强建议

为了增加数据多样性，可以使用：

1. **光照变换**
   ```python
   # 亮度调整
   brightened = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
   darkened = cv2.convertScaleAbs(frame, alpha=0.8, beta=-30)
   ```

2. **角度变换**
   ```python
   # 模拟不同手持角度
   rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
   ```

3. **添加噪声**
   ```python
   # 模拟传感器噪声
   noisy = cv2.GaussianBlur(frame, (3, 3), 0)
   ```

## 5. 数据集统计目标

| 指标 | 目标值 |
|:---|:---:|
| 总视频数 | ≥ 500 |
| 总时长 | ≥ 2 小时 |
| 场景类别数 | ≥ 10 |
| 标注完整性 | 100% |
| 难度分布均衡 | easy:medium:hard = 4:3:3 |
