"""Reproduction trainer: exact square validation, FP32 assignment, dual checkpoints."""
import copy
import hashlib
import io
import json
from datetime import datetime,timezone
from pathlib import Path
import torch
from ultralytics.data import build_yolo_dataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.checkpoints import atomic_bytes
from ultralytics.utils.torch_utils import convert_optimizer_state_dict_to_fp16
from .model import inject

ROOT=Path(__file__).resolve().parents[1]

def external(path):
    path=Path(path).expanduser().resolve()
    if path.is_relative_to(ROOT):raise ValueError('Keep generated data, weights and results OUTSIDE the source directory: '+str(path))
    return path

def write_json(path,value):
    atomic_bytes(path,(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())

class ReproValidator(DetectionValidator):
    def build_dataset(self,img_path,imgir_path,mode='val',batch=None):
        return build_yolo_dataset(self.args,img_path,imgir_path,batch,self.data,mode=mode,rect=False,stride=self.stride)
    def get_stats(self):
        stats=super().get_stats();self.raw_metrics={k:float(v) for k,v in stats.items()}
        return stats

class ReproTrainer(DetectionTrainer):
    def read_results_csv(self):
        import pandas as pd
        return {k.strip():v for k,v in pd.read_csv(self.csv).to_dict(orient='list').items()}
    def build_dataset(self,img_path,imgir_path,mode='train',batch=None):
        stride=max(int(self.model.stride.max()),32)
        return build_yolo_dataset(self.args,img_path,imgir_path,batch,self.data,mode=mode,rect=False,stride=stride)
    def get_model(self,cfg=None,weights=None,verbose=True):
        cfg=copy.deepcopy(cfg);cfg.pop('hfss_net',None)
        model=DetectionModel(cfg,nc=self.data['nc'],verbose=verbose);inject(model)
        if weights is not None:model.load(weights,verbose=verbose)
        return model
    def get_validator(self):
        self.loss_names=('box_loss','cls_loss','dfl_loss','seg_loss')
        return ReproValidator(self.test_loader,save_dir=self.save_dir,args=copy.copy(self.args),_callbacks=self.callbacks)
    def validate(self):
        rounded=self.validator(self);raw=dict(self.validator.raw_metrics)
        fitness=raw.pop('fitness',.1*raw['metrics/mAP50(B)']+.9*raw['metrics/mAP50-95(B)'])
        self._validated_epoch=self.epoch
        if self.best_fitness is None or fitness>self.best_fitness:self.best_fitness=fitness
        rounded.update(raw);rounded.pop('fitness',None)
        return rounded,fitness
    def save_model(self):
        fresh=getattr(self,'_validated_epoch',-1)==self.epoch
        if fresh:self.best_map50_value=max(getattr(self,'best_map50_value',-1.),float(self.metrics['metrics/mAP50(B)']))
        checkpoint=dict(epoch=self.epoch,best_fitness=self.best_fitness if self.best_fitness is not None else 0.,
            model=None,ema=copy.deepcopy(self.ema.ema).half(),updates=self.ema.updates,
            optimizer=convert_optimizer_state_dict_to_fp16(copy.deepcopy(self.optimizer.state_dict())),
            train_args=vars(self.args),train_metrics={**self.metrics,'fitness':float(self.fitness or 0.)},
            train_results=self.read_results_csv(),version='8.1.47',date=datetime.now(timezone.utc).isoformat(),
            best_map50_value=getattr(self,'best_map50_value',-1.),reproduction_protocol=self.release_protocol)
        buffer=io.BytesIO();torch.save(checkpoint,buffer);payload=buffer.getvalue()
        atomic_bytes(self.last,payload)
        if not fresh:return
        score=float(self.metrics['metrics/mAP50(B)'])
        if self.fitness>=self.best_fitness:atomic_bytes(self.best,payload)
        if score>=self.best_map50_value:atomic_bytes(self.wdir/'best_map50.pt',payload)
    def final_eval(self):pass  # CLI explicitly reloads both choices and evaluates in FP32.

def evaluate(weights,data,device,output,conf,iou,workers=2,split='val'):
    validator=ReproValidator(args=dict(data=str(data),split=split,device=device,imgsz=640,rect=False,batch=8,
        workers=workers,half=False,conf=conf,iou=iou,max_det=300,plots=False,save_json=False,
        project=str(output),name='validation',exist_ok=True))
    validator(model=str(weights))
    return dict(checkpoint=str(Path(weights).resolve()),sha256=hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
        metrics=validator.raw_metrics,fp32=True,imgsz=640,conf=conf,iou=iou,split=split)
