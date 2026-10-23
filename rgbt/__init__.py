"""HFSS-Net RGB-T object detection."""
import os
from pathlib import Path
os.environ.setdefault("YOLO_AUTOINSTALL","false")
os.environ.setdefault("YOLO_CONFIG_DIR",str(Path.home()/".cache/hfss-net"))
