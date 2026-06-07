"""
D435 深度摄像头模块
封装 RealSense D435 的初始化、图像获取、3D坐标转换
"""

import pyrealsense2 as rs
import numpy as np


class D435Camera:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self.profile = self.pipeline.start(self.config)
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)

        self.intrinsics = None

        # 深度滤波器链
        self.hole_filling = rs.hole_filling_filter()
        self.spatial = rs.spatial_filter()      # 空间平滑
        self.temporal = rs.temporal_filter()    # 时间平均

        # 设置硬件参数: 高精度模式
        try:
            depth_sensor = self.profile.get_device().first_depth_sensor()
            if depth_sensor.supports(rs.option.laser_power):
                depth_sensor.set_option(rs.option.laser_power, 360)  # 激光最大
            if depth_sensor.supports(rs.option.visual_preset):
                depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy
        except Exception as e:
            print(f"硬件参数设置跳过: {e}")

        print(f"D435 初始化完成 | 分辨率:{width}x{height} 深度标尺:{self.depth_scale}")

    def get_frames(self):
        """获取对齐的彩色帧和深度帧"""
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None, None

        if self.intrinsics is None:
            self.intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

        # 滤波链: 空间平滑 → 时间平均 → 空洞填充
        filtered = self.spatial.process(depth_frame)
        filtered = self.temporal.process(filtered)
        filtered = self.hole_filling.process(filtered)

        # 保留原始depth_frame用于get_3d_point
        raw_depth_frame = depth_frame

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(filtered.get_data())

        return color_image, depth_image, raw_depth_frame

    def get_3d_point(self, x, y, depth_frame):
        """像素坐标转3D坐标(米), 取周围区域中值更稳定"""
        if self.intrinsics is None:
            return None

        x, y = int(x), int(y)

        # 取周围5x5区域的中值深度，比单像素准
        depths = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                d = depth_frame.get_distance(x + dx, y + dy)
                if d > 0:
                    depths.append(d)

        if not depths:
            return None

        depth = sorted(depths)[len(depths) // 2]  # 中值

        point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], depth)
        return {'x': point[0], 'y': point[1], 'z': point[2], 'distance': depth}

    def get_depth_colormap(self, depth_image):
        """深度图转彩色可视化"""
        return rs.colorizer().colorize(
            rs.frame(depth_image)
        ).get_data() if hasattr(rs, 'colorizer') else None

    def stop(self):
        self.pipeline.stop()
        print("D435 已停止")
