"""Train SDGW + EFC + HFP for RGB-T object detection."""
import argparse
import hashlib
import platform
from pathlib import Path
import rgbt
import torch
import yaml
from rgbt.config import SETTINGS
from rgbt.model import initialize
from rgbt.runtime import ReproTrainer,evaluate,external,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=SETTINGS,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--pretrained',type=Path,help='Official COCO yolov8n.pt; required for a cold start')
    p.add_argument('--resume',type=Path,help='last.pt; preserves original total epochs/LR plan')
    p.add_argument('--project',type=Path,required=True);p.add_argument('--name',default='hfss_net')
    p.add_argument('--epochs',type=int);p.add_argument('--patience',type=int);p.add_argument('--close-mosaic',type=int)
    p.add_argument('--batch',type=int,default=32);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='0');p.add_argument('--workers',type=int,default=4);p.add_argument('--no-amp',action='store_true')
    a=p.parse_args();project=external(a.project);run=external(project/a.name)
    spec=SETTINGS[a.dataset];data=yaml.safe_load(a.data.read_text(encoding='utf-8'))
    if list(data['names'].values())!=list(spec['names'].values()):p.error('Dataset class names/order do not match --dataset')
    epochs=a.epochs if a.epochs is not None else 400
    patience=a.patience if a.patience is not None else 50
    close=a.close_mosaic if a.close_mosaic is not None else 10
    if epochs<=0 or patience<0 or not 0<=close<=epochs:p.error('Invalid epoch/patience/mosaic budget')
    import ultralytics.engine.trainer as engine
    engine.check_amp=lambda model:True
    if a.resume:
        ckpt=torch.load(a.resume,map_location='cpu',weights_only=False)
        protocol=ckpt['reproduction_protocol']
        if protocol['dataset']!=a.dataset:p.error('Resume dataset differs from the stored run')
        if any(x is not None for x in (a.epochs,a.patience,a.close_mosaic)):p.error('Resume restores the original budget; do not change epochs/patience/mosaic')
        run=Path(ckpt['train_args']['project'])/ckpt['train_args']['name'];external(run)
        if ckpt.get('format')!='sdgw-efc-hfp-v1':p.error('Use last.pt from this SDGW+EFC+HFP release; historical checkpoints need their original runtime')
        overrides=dict(ckpt['train_args']);overrides.update(device=a.device,data=str(a.data.resolve()),resume=False,exist_ok=True)
        trainer=ReproTrainer(overrides=overrides);trainer.recovery=ckpt
        trainer.model=trainer.get_model(cfg=ckpt['ema'].yaml,weights=None,verbose=False)
    else:
        if not a.pretrained or not a.pretrained.is_file():p.error('Supply the official yolov8n.pt with --pretrained')
        if run.exists():p.error('Run already exists; choose another --name or use --resume')
        torch.set_num_threads(2)
        model=initialize(a.pretrained,len(spec['names']))
        protocol=dict(dataset=a.dataset,mode='formal',epochs=epochs,patience=patience,close_mosaic=close,
            seed=a.seed,batch=a.batch,nbs=64,amp=not a.no_amp,assignment_precision='float32',imgsz=640,combination='combo__sdgw__efc__hfp',workers=a.workers,loader='epoch-bounded-v1',
            init_seed=0,pretrained_sha256=hashlib.sha256(a.pretrained.read_bytes()).hexdigest(),
            environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda),
            checkpoint_selection='best_map50.pt primary; fitness best.pt separate')
        import ultralytics.engine.trainer as engine
        engine.check_amp=lambda model:True  # No unrelated diagnostic model download.
        trainer=ReproTrainer(overrides=dict(model='hfss-net.yaml',data=str(a.data.resolve()),epochs=epochs,
            patience=patience,close_mosaic=close,batch=a.batch,nbs=64,imgsz=640,optimizer='SGD',lr0=.005,
            warmup_epochs=3.,lrf=.01,momentum=.937,weight_decay=.0005,warmup_momentum=.8,warmup_bias_lr=.1,cache=False,seg=0.,conf=spec['conf'],iou=spec['iou'],max_det=300,seed=a.seed,
            deterministic=True,pretrained=False,device=a.device,workers=a.workers,amp=not a.no_amp,
            rect=False,val=True,plots=False,save_period=-1,project=str(project),name=a.name))
        # Match the established Trainer rebuild after its seed reset; preserve
        # all initialized tensors while consuming the same constructor RNG.
        trainer.model=trainer.get_model(cfg=model.yaml,weights=model,verbose=False)
        actual=trainer.model.state_dict()
        for key,value in model.state_dict().items():torch.testing.assert_close(actual[key],value,rtol=0,atol=0)
    trainer.release_protocol=protocol
    def setup(t):
        assert t.model.model[-1].nc==len(spec['names']) and not t.model.cmbm_enabled and t.args.seg==0
        t.restore_after_setup()
        write_json(t.save_dir/'protocol.json',protocol)
    def epoch_start(t):
        t.args.val=True
    def finite(t):
        if not torch.isfinite(t.loss).all():raise FloatingPointError('Nonfinite training loss')
    trainer.add_callback('on_pretrain_routine_end',setup)
    trainer.add_callback('on_train_epoch_start',epoch_start)
    trainer.add_callback('on_train_batch_end',finite)
    trainer.add_callback('on_fit_epoch_end',lambda t:t.persist())
    trainer.train()
    results={}
    for selection,filename in [('map50','best_map50.pt'),('fitness','best.pt')]:
        results[selection]=evaluate(trainer.wdir/filename,a.data,a.device,trainer.save_dir/selection,
            spec['conf'],spec['iou'],a.workers)
    write_json(trainer.save_dir/'evaluation.json',results)
    print(results)

if __name__=='__main__':main()
