"""
武器头识别与抓取定位系统
D435深度相机 + YOLO检测武器头类型和抓取点 → 获取3D坐标 → 串口发送

流程:
  1. YOLO检测: 识别quan(拳头)和zhua(抓取点)
  2. 配对: 找到距离最近的zhua-quan对
  3. 深度: 读取zhua的3D坐标(抓取目标)
  4. 串口: 发送类型+坐标给机械臂
"""

import cv2
import json
import time
import asyncio
import pyrealsense2 as rs
from d435_camera import D435Camera
from detector import TipDetector
from serial_comm import SerialComm
from dm02_serial import DM02Serial
from anti_light import filter_detections


# ===== 配置 =====
MODEL_PATH = "model/best2.pt"
CONF = 0.5
IOU = 0.7
WIDTH = 640
HEIGHT = 480
FPS = 30

USE_SERIAL = False
SERIAL_PORT = "/dev/ttyUSB0"
SEND_INTERVAL = 0.05

# DM02 机械臂
USE_DM02 = False
DM02_PORT = "/dev/ttyACM0"
TARGET_CLASSES = [
    'WQ',           # 武器头/抓取点 (武馆)
    # 'R_R1',       # R1 KFS (梅林)
    # 'T03','T04','T05','T06','T07','T08','T09','T10','T11',
    # 'T12','T13','T14','T15','T16','T17',    # R2 KFS (梅林)
]

# 抗灯光
MIN_VARIANCE = 100


async def async_detect(detector, frame):
    return detector.detect(frame)


def find_tip_grab_pairs(zhua_list, quan_list, max_dist=200):
    """
    配对zhua(抓取点)和quan(拳头)
    规则: 每个quan找距离最近的zhua, 且距离不超过max_dist像素
    返回: [(quan_det, zhua_det), ...]
    """
    pairs = []
    used_zhua = set()

    for quan in quan_list:
        qx, qy = quan['center']
        best_dist = max_dist
        best_zhua = None

        for i, zhua in enumerate(zhua_list):
            if i in used_zhua:
                continue
            zx, zy = zhua['center']
            dist = ((qx - zx)**2 + (qy - zy)**2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_zhua = i

        if best_zhua is not None:
            pairs.append((quan, zhua_list[best_zhua]))
            used_zhua.add(best_zhua)

    return pairs


def main():
    # 初始化
    camera = D435Camera(width=WIDTH, height=HEIGHT, fps=FPS)
    detector = TipDetector(model_path=MODEL_PATH, conf=CONF, iou=IOU)

    ser = None
    if USE_SERIAL:
        ser = SerialComm()
        ser.list_ports()
        if not ser.open(port=SERIAL_PORT):
            ser = None

    dm02 = None
    if USE_DM02:
        dm02 = DM02Serial(port=DM02_PORT)
        if not dm02.open():
            print("DM02 串口打开失败，继续运行(无DM02模式)")
            dm02 = None

    intrinsics_saved = False

    print("\n按 Q 退出 | 按 S 保存截图")
    print("=" * 50)

    last_send_time = 0
    last_heartbeat = 0

    try:
        while True:
            # 采集
            color_image, depth_image, depth_frame = camera.get_frames()
            if color_image is None:
                continue

            # 首帧保存内参
            if not intrinsics_saved and camera.intrinsics:
                intr = camera.intrinsics
                params = {'fx': intr.fx, 'fy': intr.fy, 'ppx': intr.ppx, 'ppy': intr.ppy,
                          'width': intr.width, 'height': intr.height, 'depth_scale': camera.depth_scale}
                with open('intrinsics.json', 'w') as f:
                    json.dump(params, f, indent=2)
                print("内参已保存到 intrinsics.json")
                intrinsics_saved = True

            # YOLO检测
            all_detections, annotated = asyncio.run(async_detect(detector, color_image))

            # 抗灯光过滤 + 目标类别过滤
            all_detections = filter_detections(color_image, all_detections, min_variance=MIN_VARIANCE)
            target_dets = [d for d in all_detections if d['class_name'] in TARGET_CLASSES]

            # 深度图可视化
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            if target_dets:
                # 选面积最大的
                best = max(target_dets, key=lambda d: (d['bbox'][2]-d['bbox'][0]) * (d['bbox'][3]-d['bbox'][1]))
                cx, cy = best['center']
                cls_name = best['class_name']
                conf = best['confidence']

                grab_3d = camera.get_3d_point(cx, cy, depth_frame)

                cv2.circle(annotated, (cx, cy), 6, (0, 0, 255), -1)
                label = f"{cls_name} {conf:.0%}"
                cv2.putText(annotated, label, (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                if grab_3d:
                    info = f"({grab_3d['x']:.3f}, {grab_3d['y']:.3f}, {grab_3d['z']:.3f})m"
                    cv2.putText(annotated, info, (cx + 10, cy + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    cv2.circle(depth_colormap, (cx, cy), 6, (0, 255, 255), -1)
                    print(f"\r[{cls_name}] ({grab_3d['x']:.3f},{grab_3d['y']:.3f},{grab_3d['z']:.3f})m "
                          f"距离:{grab_3d['distance']:.3f}m  conf:{conf:.0%}", end="")

                    # DM02 发送
                    if dm02:
                        x_mm = round(grab_3d['x'] * 1000, 1)
                        y_mm = round(grab_3d['y'] * 1000, 1)
                        z_mm = round(grab_3d['z'] * 1000, 1)
                        dm02.send_position_only(x_mm, y_mm, z_mm, class_id=best['class_id'])

                    # 串口发送
                    if ser and time.time() - last_send_time > SEND_INTERVAL:
                        ser.send_grab_data(cls_name, grab_3d)
                        last_send_time = time.time()
                else:
                    print(f"\r[{cls_name}] 像素:({cx},{cy}) 深度无效", end="")
            else:
                print(f"\r未检测到目标", end="")
                # 心跳: 每秒发一次
                if dm02 and time.time() - last_heartbeat > 1.0:
                    dm02.send_heartbeat()
                    last_heartbeat = time.time()

            # 显示
            cv2.imshow("Tip Detect", annotated)
            cv2.imshow("Depth", depth_colormap)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("screenshot.jpg", annotated)
                print("\n已保存截图")

    except KeyboardInterrupt:
        print("\n用户中断")

    finally:
        if ser:
            ser.send_grab_data(None, None)
            ser.close()
        if dm02:
            dm02.send_stop()
            dm02.close()
        camera.stop()
        cv2.destroyAllWindows()
        print("\n退出")


if __name__ == "__main__":
    main()
