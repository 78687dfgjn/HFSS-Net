"""Reproduction trainer: exact square validation, FP32 assignment, dual checkpoints."""
import copy
import hashlib
import io
import json
import random
import numpy as np
from torch.utils.data import DataLoader
from ultralytics.data.build import seed_worker
from datetime import datetime,timezone
from pathlib import Path
import torch
from ultralytics.data import build_yolo_dataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.checkpoints import atomic_bytes
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

class EpochLoader(DataLoader):
    def reset(self):pass

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())

def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])

def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

class ReproTrainer(DetectionTrainer):
    recovery=None
    def get_dataloader(self,dataset_path,datasetir_path,batch_size=32,rank=0,mode='train'):
        native=super().get_dataloader(dataset_path,datasetir_path,batch_size=self.args.batch,rank=rank,mode=mode)
        if hasattr(native.iterator,'_shutdown_workers'):native.iterator._shutdown_workers()
        generator=torch.Generator().manual_seed(6148914691236517204+self.args.seed)
        return EpochLoader(native.dataset,batch_size=self.args.batch,shuffle=mode=='train',
            num_workers=self.args.workers if mode=='train' else self.args.workers*2,
            collate_fn=native.collate_fn,pin_memory=True,worker_init_fn=seed_worker,generator=generator,persistent_workers=False)
    def resume_training(self,ckpt):
        r=self.recovery
        if not r:return
        self.model.load_state_dict(r['model_state'],strict=True)
        self.ema.ema.load_state_dict(r['ema_state'],strict=True);self.ema.updates=r['updates']
        self.optimizer.load_state_dict(r['optimizer']);self.scaler.load_state_dict(r['scaler'])
        self.start_epoch=r['epoch']+1;self.best_fitness=r['best_fitness']
        self.stopper.__dict__.update(r['stopper']);self._last_opt_step=r['last_opt_step']
        self.best_map50=r['best_map50'];self.best_fitness_record=r['best_fitness_record']
    def restore_after_setup(self):
        r=self.recovery
        if r:
            self.scheduler.load_state_dict(r['scheduler'])
            self.train_loader.generator.set_state(r['train_generator']);self.test_loader.generator.set_state(r['val_generator'])
            restore_rng(r['rng'])
            assert [g['lr'] for g in self.optimizer.param_groups]==r['lr']
            assert self.start_epoch==r['epoch']+1 and self.stopper.__dict__==r['stopper']
            write_json(self.save_dir/f'resume-{self.start_epoch:04d}.json',dict(epoch=self.start_epoch,lr=r['lr'],stopper=dict(self.stopper.__dict__)))

    def read_results_csv(self):
        import pandas as pd
        return {k.strip():v for k,v in pd.read_csv(self.csv).to_dict(orient='list').items()}
    def build_dataset(self,img_path,imgir_path,mode='train',batch=None):
        stride=max(int(self.model.stride.max()),32)
        return build_yolo_dataset(self.args,img_path,imgir_path,batch,self.data,mode=mode,rect=False,stride=stride)
    def get_model(self,cfg=None,weights=None,verbose=True):
        cfg=copy.deepcopy(cfg);cfg.pop('release_combination',None)
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
    def save_model(self):pass  # Serialize after all epoch callbacks, before the next RNG draw.
    def persist(self):
        score=float(self.metrics['metrics/mAP50(B)']);epoch=self.epoch+1
        map_improved=not getattr(self,'best_map50',None) or score>self.best_map50['map50']
        fit_improved=not getattr(self,'best_fitness_record',None) or self.fitness>self.best_fitness_record['fitness']
        if map_improved:self.best_map50=dict(epoch=epoch,map50=score)
        if fit_improved:self.best_fitness_record=dict(epoch=epoch,fitness=float(self.fitness))
        checkpoint=dict(epoch=self.epoch,model=None,ema=copy.deepcopy(self.ema.ema).cpu(),updates=self.ema.updates,
            train_args=vars(self.args),train_metrics=dict(self.metrics),best_fitness=self.best_fitness,
            version='8.2.0',date=datetime.now(timezone.utc).isoformat(),reproduction_protocol=self.release_protocol)
        for filename,needed in [('best_map50.pt',map_improved),('best.pt',fit_improved)]:
            if needed:
                buffer=io.BytesIO();torch.save(checkpoint,buffer);atomic_bytes(self.wdir/filename,buffer.getvalue())
        checkpoint.update(model_state=cpu_state(self.model),ema_state=cpu_state(self.ema.ema),
            optimizer=self.optimizer.state_dict(),scaler=self.scaler.state_dict(),scheduler=self.scheduler.state_dict(),
            stopper=copy.deepcopy(self.stopper.__dict__),last_opt_step=getattr(self,'_last_opt_step',-1),
            rng=rng_state(),train_generator=self.train_loader.generator.get_state(),val_generator=self.test_loader.generator.get_state(),
            best_map50=self.best_map50,best_fitness_record=self.best_fitness_record,lr=[g['lr'] for g in self.optimizer.param_groups],format='sdgw-efc-hfp-v1')
        buffer=io.BytesIO();torch.save(checkpoint,buffer);atomic_bytes(self.last,buffer.getvalue())
        write_json(self.wdir/'best_map50.json',self.best_map50);write_json(self.wdir/'best_fitness.json',self.best_fitness_record)
    def final_eval(self):pass  # CLI explicitly reloads both choices and evaluates in FP32.

def evaluate(weights,data,device,output,conf,iou,workers=4,split='val',batch=32):
    validator=ReproValidator(args=dict(data=str(data),split=split,device=device,imgsz=640,rect=False,batch=batch,
        workers=workers,half=False,conf=conf,iou=iou,max_det=300,plots=False,save_json=False,
        project=str(output),name='validation',exist_ok=True))
    validator(model=str(weights))
    return dict(checkpoint=str(Path(weights).resolve()),sha256=hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
        metrics=validator.raw_metrics,fp32=True,imgsz=640,conf=conf,iou=iou,split=split)
