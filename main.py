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


# ===== 配置 =====
MODEL_PATH = "model/best.pt"
CONF = 0.5
IOU = 0.7
WIDTH = 640
HEIGHT = 480
FPS = 30

USE_SERIAL = False
SERIAL_PORT = "/dev/ttyUSB0"

# 串口发送间隔(秒)
SEND_INTERVAL = 0.05


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

    intrinsics_saved = False

    print("\n按 Q 退出 | 按 S 保存截图")
    print("=" * 50)

    last_send_time = 0

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
            zhua_list, quan_list, annotated = asyncio.run(async_detect(detector, color_image))

            detected_any = False

            # 单独显示zhua(抓取点) + 3D坐标
            for zhua in zhua_list:
                detected_any = True
                gx, gy = zhua['center']
                grab_3d = camera.get_3d_point(gx, gy, depth_frame)

                cv2.circle(annotated, (gx, gy), 6, (0, 0, 255), -1)
                cv2.putText(annotated, "zhua", (gx + 10, gy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                if grab_3d:
                    info = f"({grab_3d['x']:.3f}, {grab_3d['y']:.3f}, {grab_3d['z']:.3f})m"
                    cv2.putText(annotated, info, (gx + 10, gy + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    print(f"\r[zhua] 抓取点: ({grab_3d['x']:.3f}, {grab_3d['y']:.3f}, {grab_3d['z']:.3f})m "
                          f"距离:{grab_3d['distance']:.3f}m", end="")
                else:
                    print(f"\r[zhua] 像素:({gx},{gy}) 深度无效", end="")

            # 单独显示quan(武器头)
            for quan in quan_list:
                detected_any = True
                qx, qy = quan['center']
                cv2.putText(annotated, "quan", (qx + 10, qy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                print(f"\r[quan] 像素:({qx},{qy}) 置信度:{quan['confidence']:.0%}", end="")

            # 配对: quan+zhua都有时画配对线
            pairs = find_tip_grab_pairs(zhua_list, quan_list)
            for quan_det, zhua_det in pairs:
                cv2.line(annotated, quan_det['center'], zhua_det['center'], (0, 255, 0), 1)

            # 串口: 有配对时发送抓取数据
            if pairs:
                gx, gy = pairs[0][1]['center']
                grab_3d = camera.get_3d_point(gx, gy, depth_frame)
                if grab_3d and ser and time.time() - last_send_time > SEND_INTERVAL:
                    ser.send_grab_data('quan', grab_3d)
                    last_send_time = time.time()
            elif ser:
                ser.send_grab_data(None, None)

            # 全都没检测到
            if not detected_any:
                print(f"\r未检测到目标", end="")

            # 深度图可视化
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # 在深度图上标抓取点
            for zhua in zhua_list:
                gx, gy = zhua['center']
                cv2.circle(depth_colormap, (gx, gy), 6, (0, 255, 255), -1)
                d = depth_image[gy, gx] * camera.depth_scale
                cv2.putText(depth_colormap, f"{d:.3f}m", (gx + 10, gy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

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
        camera.stop()
        cv2.destroyAllWindows()
        print("\n退出")


if __name__ == "__main__":
    main()
