"""
YOLO 目标检测模块
识别武器头类型(quan)和抓取点(zhua)
"""

from ultralytics import YOLO
import torch


class TipDetector:
    def __init__(self, model_path="model/best.pt", conf=0.5, iou=0.7):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path).to(self.device)
        self.conf = conf
        self.iou = iou

        print(f"模型加载完成 | 设备:{self.device} | 类别:{self.model.names}")

    def detect(self, frame):
        """返回所有检测结果
            all_detections: [{'bbox':.., 'center':(cx,cy), 'confidence':.., 'class_name':.., 'class_id':..}]
            annotated: 标注后的图像
        """
        results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)

        all_detections = []
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = r.names[cls_id]

                x1, y1, x2, y2 = xyxy
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                all_detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'center': (cx, cy),
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': cls_name
                })

        annotated = results[0].plot()
        return all_detections, annotated
