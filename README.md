# 武器头识别与抓取定位系统

D435深度相机 + YOLO检测武器头类型和抓取点，获取3D坐标，串口发送给机械臂。

## 功能

- YOLO实时检测武器头类型(quan)和抓取点(zhua)
- 自动配对：quan下方最近的zhua作为抓取目标
- D435深度图获取抓取点3D坐标(米)
- 串口发送类型+坐标给机械臂控制器

## 文件结构

```
tip_detect_project/
├── main.py              # 主程序
├── d435_camera.py       # D435摄像头模块
├── detector.py          # YOLO检测模块
├── serial_comm.py       # 串口通讯模块
├── model/
│   └── best.pt          # 训练好的YOLO模型(2类:quan,zhua)
├── requirements.txt
└── README.md
```

## 模型类别

| 类ID | 名称 | 说明 |
|------|------|------|
| 0 | quan | 拳头型武器头 |
| 1 | zhua | 抓取点(圆柱) |

## 使用

```bash
cd ~/桌面/model_wuqi/tip_detect_project
source ~/yolo-project/.venv/bin/activate
python main.py
```

按 Q 退出，按 S 保存截图。

## 配置

main.py 顶部：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `MODEL_PATH` | 模型路径 | `model/best.pt` |
| `CONF` | 置信度阈值 | 0.5 |
| `USE_SERIAL` | 是否启用串口 | False |
| `SERIAL_PORT` | 串口路径 | `/dev/ttyUSB0` |

## 串口数据帧格式 (11字节)

```
[0xAA][类型][xH][xL][yH][yL][zH][zL][距离H][距离L][0x55]

类型: 0x00=无目标  0x01=quan
x/y/z: float转mm + 32768偏移(处理负数), 16位
距离: mm, 16位
```

## 扩展

后续训练矛尖(maojian)和掌(zhang)类别后，只需在 model/best.pt 替换模型，
detector.py 已支持自动读取模型类别。
# d435_weapon
