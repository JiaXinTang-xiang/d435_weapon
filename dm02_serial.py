"""
DM02 六轴机械臂 USB CDC 串口通信
协议: 34 字节固定帧, 参考遥控器 SBUS 模式

帧格式:
  0     1      2-5    6-9   10-13  14-17  18-21  22-25  26-29    30       31       32
 0xAA  CMD   X(mm)  Y(mm)  Z(mm)   qw     qx     qy     qz    gripper   CRC   class_id
"""

import struct
import serial
import time
import numpy as np


# ===== 协议常量 =====
FRAME_HEADER = 0xAA
FRAME_SIZE   = 34

# 命令
CMD_MOVE      = 0x01   # 移动到目标位姿 (抓取)
CMD_STOP      = 0x02   # 急停
CMD_HOME      = 0x03   # 回零
CMD_HEARTBEAT = 0x04   # 心跳 (上位机存活信号)

# 夹爪
GRIPPER_NONE   = 0  # 不动
GRIPPER_CLOSE  = 1  # 闭合
GRIPPER_OPEN   = 2  # 张开

# ===== 固定抓取姿态 (水平朝前抓) =====
# 四元数 (qw, qx, qy, qz), 需要根据实际机械臂末端朝向调整
GRASP_QUAT_STOP     = (1.0, 0.0, 0.0, 0.0)   # 默认不动 (单位四元数)
GRASP_QUAT_FORWARD  = (1.0, 0.0, 0.0, 0.0)   # 正面水平抓 (待实测调整)


class DM02Serial:
    """DM02 USB CDC 串口通信"""

    def __init__(self, port="/dev/ttyACM0", baudrate=115200):
        """
        Args:
            port:     串口路径, DM02 USB CDC 通常是 /dev/ttyACM0
            baudrate: 波特率 (USB CDC 虚拟串口, 此参数通常被忽略)
        """
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self._frame = bytearray(FRAME_SIZE)

    # ==================== 帧构建 ====================

    def _build_move_frame(self, x_mm, y_mm, z_mm,
                           qw, qx, qy, qz, gripper=GRIPPER_NONE,
                           class_id=0):
        """构建移动命令帧

        Args:
            x_mm, y_mm, z_mm: 目标位置 mm (相机坐标系)
            qw,qx,qy,qz:      夹爪姿态四元数
            gripper:          夹爪控制
            class_id:         目标类别ID (0-255, 由下位机判断)

        Returns:
            34 bytes frame
        """
        # 打包: 帧头+CMD+6浮点+gripper+CRC+class_id = 29 bytes, 补齐到34
        data = struct.pack('<BBffffffBBB',
                           FRAME_HEADER,
                           CMD_MOVE,
                           x_mm, y_mm, z_mm,
                           qw, qx, qy, qz,
                           gripper,
                           class_id,
                           0x00)  # reserved
        return data + b'\x00' * (FRAME_SIZE - len(data))

    def _build_stop_frame(self):
        """构建急停帧"""
        data = struct.pack('<BB', FRAME_HEADER, CMD_STOP)
        return data + b'\x00' * (FRAME_SIZE - len(data))

    def _build_heartbeat_frame(self):
        """构建心跳帧"""
        data = struct.pack('<BB', FRAME_HEADER, CMD_HEARTBEAT)
        return data + b'\x00' * (FRAME_SIZE - len(data))

    # ==================== 发送接口 ====================

    def send_position(self, x_mm, y_mm, z_mm,
                       quat=GRASP_QUAT_FORWARD, class_id=0):
        """发送目标位置 + 抓取姿态 + 类别

        Args:
            x_mm, y_mm, z_mm: 目标位置 mm
            quat:             (qw, qx, qy, qz) 抓取姿态
            class_id:         目标类别ID (下位机自行判断)
        """
        frame = self._build_move_frame(
            x_mm, y_mm, z_mm,
            quat[0], quat[1], quat[2], quat[3],
            gripper=GRIPPER_NONE,
            class_id=class_id
        )
        self._write(frame)

    def send_position_only(self, x_mm, y_mm, z_mm,
                            quat=GRASP_QUAT_STOP, class_id=0):
        """仅发送目标位置, 不动夹爪 (调试用)"""
        frame = self._build_move_frame(
            x_mm, y_mm, z_mm,
            quat[0], quat[1], quat[2], quat[3],
            gripper=GRIPPER_NONE,
            class_id=class_id
        )
        self._write(frame)

    def send_gripper(self, action):
        """单独控制夹爪

        Args:
            action: GRIPPER_OPEN / GRIPPER_CLOSE
        """
        frame = self._build_move_frame(
            0, 0, 0,
            1.0, 0.0, 0.0, 0.0,
            gripper=action,
            class_id=0
        )
        self._write(frame)

    def send_stop(self):
        """发送急停"""
        self._write(self._build_stop_frame())

    def send_heartbeat(self):
        """发送心跳 (上位机存活信号)"""
        self._write(self._build_heartbeat_frame())

    def send_config(self, mode):
        """发送配置帧: 0=wuguan, 1=meilin"""
        data = struct.pack('<BBB', FRAME_HEADER, CMD_CONFIG, mode)
        self._write(data + b'\x00' * (FRAME_SIZE - len(data)))

    def _write(self, frame):
        """写串口"""
        if self.ser and self.ser.is_open:
            self.ser.write(frame)

    # ==================== 串口管理 ====================

    def open(self, port=None):
        """打开串口"""
        if port:
            self.port = port

        if self.ser and self.ser.is_open:
            return True

        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=8,
                parity='N',
                stopbits=1,
                timeout=0.1
            )
            print(f"DM02 串口已打开: {self.port} @ {self.baudrate}")
            return True
        except Exception as e:
            print(f"DM02 串口打开失败: {e}")
            self.ser = None
            return False

    def close(self):
        """关闭串口"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.ser = None
            print("DM02 串口已关闭")

    @property
    def is_open(self):
        return self.ser is not None and self.ser.is_open

    # ==================== 工具函数 ====================

    @staticmethod
    def camera_to_mm(x_m, y_m, z_m):
        """相机坐标系 (米) → 毫米

        Args:
            x_m, y_m, z_m: 来自 get_3d_point()

        Returns:
            (x_mm, y_mm, z_mm)
        """
        return (
            round(x_m * 1000, 1),
            round(y_m * 1000, 1),
            round(z_m * 1000, 1),
        )

    @staticmethod
    def euler_to_quat(roll, pitch, yaw):
        """欧拉角 (rad) → 四元数 (DM02 使用 ZYX 顺序)

        Args:
            roll, pitch, yaw: 弧度

        Returns:
            (qw, qx, qy, qz)
        """
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return (qw, qx, qy, qz)


# ===== 快速测试 =====
if __name__ == "__main__":
    dm = DM02Serial(port="/dev/ttyACM0")
    if dm.open():
        print("连接成功, 发送测试帧...")
        dm.send_position_only(100.0, -50.0, 400.0)
        dm.close()
    else:
        print("DM02 未连接, 跳过测试")
