"""
Scaled-YOLOv4 builders for the three tasks (anchor-based).

Paper : Wang, Bochkovskiy & Liao, *Scaled-YOLOv4: Scaling Cross Stage Partial Network*, CVPR 2021 (arXiv:2011.08036)
Code  : official Darknet cfg (see the cfg headers), built by the common Darknet builder.
Delta : YOLOv4-CSP - CSP applied to the whole backbone and neck (CSPDarknet53 + CSP-SPP + CSP-PAN), Mish
Detection layers: reused unchanged from detectors/yolov3 (as v2 / v4 / v5 / v7)
    image -> Image YOLO layer, complex -> Complex-YOLO layer, yolo3d -> YOLO3D layer
Not used (same policy as v4 / v7): scale_x_y grid sensitivity, CIoU, Mosaic and the other bag-of-freebies.
"""

from pathlib import Path

from common.darknet import Darknet
from detectors.yolov3.build import yolo3d_factory

CFG_DIR = Path(__file__).resolve().parent / "cfg"
SPECS = {
    "yolo3d": ("yolo3d_scaledyolov4.cfg", yolo3d_factory, 8),
}


def build_model(task, num_classes):
    cfg_name, factory, num_box_attr = SPECS[task]
    return Darknet(CFG_DIR / cfg_name, factory, num_classes, num_box_attr)
