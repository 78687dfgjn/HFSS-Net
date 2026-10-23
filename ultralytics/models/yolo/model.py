# Detection-only adapter; Ultralytics-derived, AGPL-3.0.
from ultralytics.engine.model import Model
from ultralytics.nn.tasks import DetectionModel
from .detect import DetectionTrainer,DetectionValidator,DetectionPredictor
class YOLO(Model):
    @property
    def task_map(self):
        return {"detect":dict(model=DetectionModel,trainer=DetectionTrainer,
            validator=DetectionValidator,predictor=DetectionPredictor)}
