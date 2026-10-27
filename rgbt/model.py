"""SDGW + EFC + DCT-HFP, matching combo__sdgw__efc__hfp slot placement."""
import copy,math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors
from .config import ARCHITECTURE
from .sdgw import SDGW
from .efc import EFC

def fp32_context(x):return torch.autocast(device_type=x.device.type,enabled=False)

class HFP(nn.Module):
    def __init__(self,channels,ratio=(.25,.25),patch=(8,8)):
        super().__init__();assert channels%32==0
        self.ratio,self.patch=ratio,patch
        self.channel1=nn.Conv2d(channels,channels,1,groups=32,bias=False)
        self.channel2=nn.Conv2d(channels,channels,1,groups=32,bias=False)
        self.out=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1),nn.GroupNorm(32,channels))
        self.register_buffer('_basis_h',torch.empty(0),persistent=False)
        self.register_buffer('_basis_w',torch.empty(0),persistent=False)

    @staticmethod
    def basis(size,device):
        k=torch.arange(size,device=device,dtype=torch.float32)[:,None]
        n=torch.arange(size,device=device,dtype=torch.float32)[None,:]
        matrix=torch.cos(math.pi/size*(n+.5)*k)*math.sqrt(2/size)
        matrix[0]*=1/math.sqrt(2)
        return matrix

    def high_pass(self,x):
        h,w=x.shape[-2:]
        if self._basis_h.shape!=(h,h) or self._basis_h.dtype!=torch.float32:
            self._basis_h=self.basis(h,x.device)
        if self._basis_w.shape!=(w,w) or self._basis_w.dtype!=torch.float32:
            self._basis_w=self.basis(w,x.device)
        a,b=self._basis_h,self._basis_w
        coeff=torch.matmul(torch.matmul(a,x),b.T)
        mask=torch.ones((h,w),device=x.device,dtype=torch.float32)
        mask[:int(h*self.ratio[0]),:int(w*self.ratio[1])]=0
        return torch.matmul(torch.matmul(a.T,coeff*mask),b)

    def forward(self,x):
        dtype=x.dtype
        with fp32_context(x):
            z=x.float();high=self.high_pass(z)
            maximum=F.adaptive_max_pool2d(high,self.patch).relu().sum((2,3),keepdim=True)
            average=F.adaptive_avg_pool2d(high,self.patch).relu().sum((2,3),keepdim=True)
            attention=torch.sigmoid(self.channel2(self.channel1(maximum)+self.channel1(average)))
            result=self.out(z*high+z*attention)
        return result.to(dtype)



class AddFusion(nn.Module):
    is_cross_modal_fusion=True
    def forward(self,x):return x[2]+x[3]

class ZeroSegmentationHead(nn.Module):
    def __init__(self,classes):super().__init__();self.classes=classes
    def forward(self,x):return x.new_zeros((x.shape[0],self.classes,*x.shape[-2:]))

def route(new,old):
    for key in ('f','i'):setattr(new,key,getattr(old,key))
    new.type=type(new).__module__+'.'+type(new).__name__
    new.np=sum(p.numel() for p in new.parameters());return new

class Residual(nn.Module):
    def __init__(self,original,name,channels):
        super().__init__();self.orig=original
        self.module=SDGW(channels) if name=='sdgw' else HFP(channels)
        self.gamma=nn.Parameter(torch.zeros(()));route(self,original)
    def forward(self,x):
        base=self.orig(x)
        with fp32_context(base):extra=self.module(base.float())
        if extra.shape!=base.shape:raise ValueError('Single-slot shape changed')
        return base+self.gamma*extra.to(base.dtype)

class PairModule(nn.Module):
    def __init__(self,channels):
        super().__init__();self.name='efc';self.module=EFC(channels,channels)
    def forward(self,rgb,tir):return self.module([rgb,tir])

class Fusion(nn.Module):
    is_cross_modal_fusion=True
    def __init__(self,original,channels):
        super().__init__();self.module=PairModule(channels)
        self.gamma=nn.Parameter(torch.zeros(()));route(self,original)
    def forward(self,x):
        rgb,tir=x[2],x[3];base=rgb+tir
        with fp32_context(base):extra=self.module(rgb.float(),tir.float())
        if extra.shape!=base.shape:raise ValueError('Fusion shape changed')
        return base+self.gamma*extra.to(base.dtype)

class FP32Assigner(nn.Module):
    def __init__(self,original,owner):super().__init__();self.original=original;self.owner=owner
    @torch.no_grad()
    def forward(self,scores,boxes,points,labels,gt_boxes,mask):
        owner=self.owner;raw,distribution,anchor,stride=owner.assignment_cache
        with torch.autocast(device_type=raw.device.type,enabled=False):
            anchor,stride=anchor.float(),stride.float()
            return self.original(raw.float().sigmoid(),owner.bbox_decode(anchor,distribution.float())*stride,
                anchor*stride,labels.float(),gt_boxes.float(),mask)

class FP32AssignmentLoss(v8DetectionLoss):
    def __init__(self,model):
        super().__init__(model);self.assignment_cache=None;self.assigner=FP32Assigner(self.assigner,self)
    def __call__(self,preds,batch):
        feats=preds[1] if isinstance(preds,tuple) else preds
        with torch.no_grad():
            distribution,raw=torch.cat([x.detach().view(x.shape[0],self.no,-1) for x in feats],2).split((self.reg_max*4,self.nc),1)
            anchor,stride=make_anchors(feats,self.stride,.5)
            self.assignment_cache=(raw.permute(0,2,1).contiguous(),distribution.permute(0,2,1).contiguous(),anchor,stride)
        try:return super().__call__(preds,batch)
        finally:self.assignment_cache=None

class ReproModel(DetectionModel):
    def init_criterion(self):return FP32AssignmentLoss(self)
    def _apply(self,fn):
        if hasattr(self,'criterion'):del self.criterion
        return super()._apply(fn)

SPEC={'G1':'sdgw','G2':'efc','G3':'hfp'}

def inject(model):
    if model.yaml.get('release_combination'):
        assert model.yaml['release_combination']==SPEC;return model
    assert len(model.model)==40 and tuple(model.model[-1].f)==(32,35,38,31)
    for index in (23,24,25):model.model[index]=route(AddFusion(),model.model[index])
    positions={
        'sdgw':[(9,64),(11,64),(14,128),(16,128),(20,256),(22,256)],
        'efc':[(23,64),(24,128),(25,256)],
        'hfp':[(32,64),(35,128),(38,256)],
    }
    for name,slots in positions.items():
        for index,channels in slots:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(100000+index*100+sum((k+1)*ord(v) for k,v in enumerate(name)))
                old=model.model[index]
                model.model[index]=Fusion(old,channels) if name=='efc' else Residual(old,name,channels)
    model.model[-1].segmentation_head=ZeroSegmentationHead(model.model[-1].num_seg_classes)
    model.cmbm_enabled=False;model.yaml=copy.deepcopy(model.yaml)
    model.yaml.update(release_combination=copy.deepcopy(SPEC),paper_mode=True)
    model.__class__=ReproModel
    if hasattr(model,'criterion'):del model.criterion
    return model

BACKBONE_MAP={0:0,1:1,2:2,3:3,9:4,10:5,14:6,15:7,19:8,20:9,
              4:0,5:1,6:2,7:3,11:4,12:5,16:6,17:7,21:8,22:9}

def initialize(pretrained,nc):
    """Rebuild both historical initialization streams without distributing private checkpoints."""
    if nc not in (1,3):raise ValueError('Only the documented one/three-class datasets are supported')
    checkpoint=torch.load(Path(pretrained),map_location='cpu',weights_only=False)
    source=(checkpoint.get('ema') or checkpoint['model']).float().state_dict()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        three=DetectionModel(copy.deepcopy(ARCHITECTURE),verbose=False)
        target=three.state_dict();copied={}
        for dst,src in BACKBONE_MAP.items():
            prefix=f'model.{dst}.'
            for key,value in target.items():
                if key.startswith(prefix):
                    src_key=f'model.{src}.'+key[len(prefix):]
                    if src_key not in source or source[src_key].shape!=value.shape:raise ValueError('Incompatible official YOLOv8n backbone: '+src_key)
                    copied[key]=source[src_key].clone()
        if len(copied)!=324:raise ValueError(f'Expected324 backbone tensors, got{len(copied)}')
        three.load_state_dict(copied,strict=False)
        if nc==3:model=three
        else:
            torch.random.default_generator.manual_seed(0)
            cfg=copy.deepcopy(three.yaml);cfg.update(nc=1,ablation_group='neither',ablation_gaffm=False,ablation_psab=False)
            model=DetectionModel(cfg,verbose=False);state=model.state_dict()
            model.load_state_dict({k:v for k,v in three.state_dict().items() if k in state and state[k].shape==v.shape},strict=False)
        inject(model)
    return model
