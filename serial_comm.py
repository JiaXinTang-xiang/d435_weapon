"""
串口通讯模块
发送抓取点3D坐标和武器头类型给机械臂控制器

数据帧格式 (11字节):
  [0xAA][类型][xH][xL][yH][yL][zH][zL][距离H][距离L][0x55]

  类型: 0x00=无目标  0x01=quan(拳头)
  x/y/z: float转int(mm), 16位
  距离: mm, 16位
"""

import serial
import serial.tools.list_ports


class SerialComm:
    def __init__(self):
        self.ser = None

    def list_ports(self):
        ports = serial.tools.list_ports.comports()
        for p in ports:
            print(f"  {p.device} - {p.description}")
        return ports

    def open(self, port="/dev/ttyUSB0", baudrate=115200):
        try:
            self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=0.1)
            print(f"串口已打开: {port}")
            return True
        except Exception as e:
            print(f"串口打开失败: {e}")
            return False

    def send_grab_data(self, tip_type, grab_3d):
        """
        发送抓取数据
        tip_type: 'quan' 或 None
        grab_3d: {'x':float, 'y':float, 'z':float, 'distance':float} 米
        """
        if self.ser is None or not self.ser.is_open:
            return

        # 类型编码
        type_code = 0x01 if tip_type == 'quan' else 0x00

        # 坐标转mm (float → 16bit int, 偏移+32768处理负数)
        if grab_3d:
            x_mm = int(grab_3d['x'] * 1000) + 32768
            y_mm = int(grab_3d['y'] * 1000) + 32768
            z_mm = int(grab_3d['z'] * 1000) + 32768
            dist_mm = int(grab_3d['distance'] * 1000)
        else:
            x_mm = y_mm = z_mm = dist_mm = 0

        # 限幅
        x_mm = max(0, min(65535, x_mm))
        y_mm = max(0, min(65535, y_mm))
        z_mm = max(0, min(65535, z_mm))
        dist_mm = max(0, min(65535, dist_mm))

        # 构建帧
        packet = bytearray([
            0xAA,
            type_code,
            (x_mm >> 8) & 0xFF, x_mm & 0xFF,
            (y_mm >> 8) & 0xFF, y_mm & 0xFF,
            (z_mm >> 8) & 0xFF, z_mm & 0xFF,
            (dist_mm >> 8) & 0xFF, dist_mm & 0xFF,
            0x55
        ])
        self.ser.write(packet)

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("串口已关闭")
