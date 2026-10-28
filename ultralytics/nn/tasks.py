import contextlib
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm
import cv2
import torch.nn.functional as F

from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C3,
    C3TR,
    OBB,
    SPP,
    SPPELAN,
    SPPF,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C3Ghost,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    KANConv,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    ResNetLayer,
    RTDETRDecoder,
    Segment,
    Silence,
    WorldDetect,
    Concat2,
    ADD,
    ShuffleAttention,
    SimAM,
    GAM_Attention,
    CBAM2,
    CoordAtt,
    ECA,
    SEAttention,
    GLCBAM,
    S2Attention,
    SKAttention,
    GLF,
    NAM,
    GCBAM,
    SACBAM,
    MdC2f,
    CDC2f,
    C2f_Shufflenet,
    C2f_Invo,
    C2f_PKIModule,
    CSFM,
    FEM,
    C2f_FEM,
    C2f_PPA,
    C2f_RG,
    C2f_Faster,
    Fusion,
    Concat3,
    RIFusion,
    RIFusion1,
    FusionModule,
    Concat1,
    Concat4,
    Concat5,
    FusionModule_ch,
    CMF,
    RFAConv,
    SpaFre,
    SARC,
    FeatureAlign,
    PairedAdd,
    downsample,
    DCAF,
    CAM,
    SAM,
    SCAM,
    PConv,
    AMG,
    SpatialAttention,
)
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import v8ClassificationLoss, v8DetectionLoss, v8OBBLoss, v8PoseLoss, v8SegmentationLoss
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    make_divisible,
    model_info,
    scale_img,
    time_sync,
)

try:
    import thop
except ImportError:
    thop = None


class BaseModel(nn.Module):


    def forward(self, x, *args, **kwargs):









        if isinstance(x, dict):
            return self.loss(x,*args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):













        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)





    def _predict_once(self, x, profile=False, visualize=False, embed=None):

        if x.shape[1] != 6:
            raise ValueError(f"PTANet requires 6 RGB/thermal channels, got {x.shape[1]}")
        rgb, ir = x.chunk(2, dim=1)
        y, dt = [], []
        current, visible = rgb, True
        fusion_start = next(m.i for m in self.model if getattr(m, 'is_cross_modal_fusion', False))
        for m in self.model:
            if m.f == -4:
                visible = not visible
                current = m(rgb if visible else ir)
            elif m.f == -3:
                if any(True for _ in m.parameters()):
                    rgb, ir = m(torch.cat((rgb, ir), dim=1)).chunk(2, dim=1)
                current = rgb if visible else ir
            else:
                if m.f != -1:
                    current = y[m.f] if isinstance(m.f, int) else [current if j == -1 else y[j] for j in m.f]
                if profile:
                    self._profile_one_layer(m, current, dt)
                current = m(current)
            if m.i < fusion_start and m.f != -3:
                if visible:
                    rgb = current
                else:
                    ir = current
            y.append(current if m.i in self.save else None)
        return current

    def _predict_augment(self, x):

        LOGGER.warning(
            f"WARNING ⚠️ {self.__class__.__name__} does not support augmented inference yet. "
            f"Reverting to single-scale inference instead."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):












        c = m == self.model[-1] and isinstance(x, list)
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):







        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)
                    delattr(m, "bn")
                    m.forward = m.forward_fuse
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")
                    m.forward = m.forward_fuse
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):









        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
        return sum(isinstance(v, bn) for v in self.modules()) < thresh

    def info(self, detailed=False, verbose=True, imgsz=640):








        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):









        self = super()._apply(fn)
        m = self.model[-1]
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):







        model = weights["model"] if isinstance(weights, dict) else weights
        csd = model.float().state_dict()
        csd = intersect_dicts(csd, self.state_dict())
        self.load_state_dict(csd, strict=False)
        if verbose:
            LOGGER.info(f"Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights")

    def loss(self, batch, preds=None):







        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):

        raise NotImplementedError("compute_loss() needs to be implemented by task heads")


class DetectionModel(BaseModel):


    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):

        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)


        ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}
        self.inplace = self.yaml.get("inplace", True)



        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            m.inplace = self.inplace

            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Pose, OBB)) else self.forward(x)


            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(2, 6, s, s))[1]])
            self.stride = m.stride
            m.bias_init()
        else:
            self.stride = torch.Tensor([32])


        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")




    def _predict_augment(self, x):

        img_size = x.shape[-2:]
        s = [1, 0.83, 0.67]
        f = [None, 3, None]
        y = []
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)
        return torch.cat(y, -1), None

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):

        p[:, :4] /= scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y
        elif flips == 3:
            x = img_size[1] - x
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):

        nl = self.model[-1].nl
        g = sum(4**x for x in range(nl))
        e = 1
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))
        y[0] = y[0][..., :-i]
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))
        y[-1] = y[-1][..., i:]
        return y

    def init_criterion(self):

        return v8DetectionLoss(self)


class OBBModel(DetectionModel):


    def __init__(self, cfg="yolov8n-obb.yaml", ch=3, nc=None, verbose=True):

        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)


    def init_criterion(self):

        return v8OBBLoss(self)


class SegmentationModel(DetectionModel):


    def __init__(self, cfg="yolov8n-seg.yaml", ch=3, nc=None, verbose=True):

        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):

        return v8SegmentationLoss(self)


class PoseModel(DetectionModel):


    def __init__(self, cfg="yolov8n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):

        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):

        return v8PoseLoss(self)


class ClassificationModel(BaseModel):


    def __init__(self, cfg="yolov8n-cls.yaml", ch=3, nc=None, verbose=True):

        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):

        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)


        ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError("nc not specified. Must specify nc in model.yaml or function arguments.")
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
        self.stride = torch.Tensor([1])
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):

        name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]
        if isinstance(m, Classify):
            if m.linear.out_features != nc:
                m.linear = nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, nn.Linear):
            if m.out_features != nc:
                setattr(model, name, nn.Linear(m.in_features, nc))
        elif isinstance(m, nn.Sequential):
            types = [type(x) for x in m]
            if nn.Linear in types:
                i = types.index(nn.Linear)
                if m[i].out_features != nc:
                    m[i] = nn.Linear(m[i].in_features, nc)
            elif nn.Conv2d in types:
                i = types.index(nn.Conv2d)
                if m[i].out_channels != nc:
                    m[i] = nn.Conv2d(m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)

    def init_criterion(self):

        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):



















    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):









        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):

        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):










        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]

        bs = len(img)
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        preds = self.predict(img, batch=targets) if preds is None else preds
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )

        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):














        y, dt, embeddings = [], [], []
        for m in self.model[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            y.append(x if m.i in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)
        return x


class WorldModel(DetectionModel):


    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):

        self.txt_feats = torch.randn(1, nc or 80, 512)
        self.clip_model = None
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def set_classes(self, text, batch=80, cache_clip_model=True):

        try:
            import clip
        except ImportError:
            check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip

        if (
            not getattr(self, "clip_model", None) and cache_clip_model
        ):
            self.clip_model = clip.load("ViT-B/32")[0]
        model = self.clip_model if cache_clip_model else clip.load("ViT-B/32")[0]
        device = next(model.parameters()).device
        text_token = clip.tokenize(text).to(device)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        self.txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        self.model[-1].nc = len(text)

    def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):














        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if len(txt_feats) != len(x):
            txt_feats = txt_feats.repeat(len(x), 1, 1)
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)

            y.append(x if m.i in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):







        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
        return self.criterion(preds, batch)


class Ensemble(nn.ModuleList):


    def __init__(self):

        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):

        y = [module(x, augment, profile, visualize)[0] for module in self]


        y = torch.cat(y, 2)
        return y, None





@contextlib.contextmanager
def temporary_modules(modules=None):





















    if not modules:
        modules = {}

    import importlib
    import sys

    try:

        for old, new in modules.items():
            sys.modules[old] = importlib.import_module(new)

        yield
    finally:

        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


def torch_safe_load(weight):











    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)
    try:
        with temporary_modules(
            {
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            }
        ):
            ckpt = torch.load(file, map_location="cpu")

    except ModuleNotFoundError as e:
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                    f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'"
                )
            ) from e
        LOGGER.warning(
            f"WARNING ⚠️ {weight} appears to require '{e.name}', which is not in ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
            f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'"
        )
        check_requirements(e.name)
        ckpt = torch.load(file, map_location="cpu")

    if not isinstance(ckpt, dict):

        LOGGER.warning(
            f"WARNING ⚠️ The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):


    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)
        args = {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()


        model.args = args
        model.pt_path = w
        model.task = guess_model_task(model)
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])


        ensemble.append(model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval())


    for m in ensemble.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None


    if len(ensemble) == 1:
        return ensemble[-1]


    LOGGER.info(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f"Models differ in class counts {[m.nc for m in ensemble]}"
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):

    ckpt, weight = torch_safe_load(weight)
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}
    model = (ckpt.get("ema") or ckpt["model"]).to(device).float()


    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}
    model.pt_path = weight
    model.task = guess_model_task(model)
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()


    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None


    return model, ckpt


def parse_model(d, ch, verbose=True):

    import ast


    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")

    ch = [ch]

    tx=[3,256,256,512,512,1024,1024]
    tz = [64,128,256]


    ty=0
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):








        try:
            if m == 'node_mode':
                m = d[m]
                if len(args) > 0:
                    if args[0] == 'head_channel':
                        args[0] = int(d[args[0]])
            t = m
            m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]
        except:
            pass
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n



        if m in {
            Classify,
            KANConv,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            RepNCSPELAN4,
            ADown,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            MdC2f,
            CDC2f,
            C2f_Shufflenet,
            C2f_Invo,
            C2f_PKIModule,
            C2f_FEM,
            C2f_Faster,
            C2f_RG,
            RFAConv,
            PConv,

        }:

            c1, c2 = ch[f], args[0]
            if f == -4:
                c1=tx[ty]
                if ty!=0:
                    c1=c1*width

                c1=int(c1)
                ty+=1
            if f == -2:
                c1 = tx[ty]
                c1 = int(c1)





            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(
                    max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2]
                )

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C1, C2, C2f, C2fAttn, C3, C3TR, C3Ghost, C3x, RepC3,C2f_PKIModule,C2f_FEM,C2f_PPA,C2f_RG,C2f_Faster}:
                args.insert(2, n)
                n = 1
        elif m is Conv2:
            c1, c2 = ch[f], args[0]
            if f == -4:
                c1 = tx[ty]
                if ty != 0:
                    c1 = c1 * width

                c1 = int(c1)
                ty += 1
            args = [c1, c2, *args[1:]]

        elif m is FEM:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            tn=args[1]

            args=[c1,c2,tn]
        elif m is Fusion:
            args[0] = d[args[0]]
            c1, c2 = [ch[x] for x in f], (sum([ch[x] for x in f]) if args[0] == 'concat' else ch[f[0]])
            args = [c1, args[0]]
        elif m in {ADD, FusionModule, CMF, SpaFre, SARC,DCAF,CAM,SAM,SCAM,PairedAdd}:

            c2 = ch[f[0]]
            if m is PairedAdd:
                args=[args[0],d.get('preserve_init_rng',True)]


        elif m is S2Attention:
            c1 = ch[f[0]]+ch[f[1]]
            c2 = ch[f[0]]
            args = [c1, c2]



        elif m in {RIFusion, RIFusion1, FusionModule_ch,FeatureAlign,downsample,AMG,SpatialAttention}:
            args = [args[0]]
        elif m in {SKAttention,GLF,NAM,GLCBAM,GCBAM,SACBAM,CSFM}:
            c1 = ch[f[0]]+ch[f[1]]
            c2 = ch[f[0]]
            args = [c1,c2]
        elif m is ShuffleAttention:
            c1 = ch[f[0]]+ch[f[1]]
            c2 = ch[f[0]]
            args = [c1,c2]
        elif m is SimAM:

            c1 = ch[f[0]]+ch[f[1]]
            c2 = ch[f[0]]
            args = [c1,c2]
        elif m is GAM_Attention:
            c1 = ch[f[0]]+ch[f[1]]
            c2 = ch[f[0]]
            args = [c1,c2]
        elif m is Concat1:
            c2 = ch[f[0]] + tz[1]
        elif m is Concat4:
            c2 = ch[f[0]] + tz[0]
        elif m is Concat5:
            c2 = ch[f[0]] + tz[2]
        elif m is Concat2:

            c1 = ch[f[0]]+ch[f[1]]
            c2 = ch[f[0]]
            args = [c1,c2]
        elif m is Concat3:

            c1 = ch[f[0]]+ch[f[1]]
            c2 = ch[f[0]]
            args = [c1,c2]
        elif m is SEAttention:
            c1 = ch[f[0]]+ch[f[1]]
            c2 = ch[f[0]]
            args = [c1,c2]
        elif m is ECA:
            c1 = ch[f[0]]+ch[f[1]]
            c2 = ch[f[0]]
            args = [c1,c2]

        elif m is CBAM2:
            c1 = ch[f[0]]+ch[f[1]]
            c2 = ch[f[0]]
            args = [c1,c2]
        elif m is CoordAtt:
            c1 = ch[f[0]]+ch[f[1]]
            c2 = ch[f[0]]
            args = [c1,c2]
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in {HGStem, HGBlock}:
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn}:
            num_seg_classes = num_seg_classes if 'num_seg_classes' in locals() else 1
            args.append([ch[x] for x in f])
            if m is Detect and d.get('paper_mode'):
                args.extend([nc, 8])
            if m is Detect and hasattr(m, 'num_seg_classes'):
                args[-1].append(num_seg_classes)
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)

        elif m is RTDETRDecoder:
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]

        else:
            c2 = ch[f]


        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)



        t = str(m)[8:-2].replace("__main__.", "")
        m.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}")

        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x not in [-1,-3,-4,-2])


        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)


    return nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):

    import re

    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)
    d["scale"] = guess_model_scale(path) or d.get('scale', 'n')
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):











    with contextlib.suppress(AttributeError):
        import re

        return re.search(r"yolov\d+([nslmx])", Path(model_path).stem).group(1)
    return ""


def guess_model_task(model):













    def cfg2task(cfg):

        m = cfg["head"][-1][-2].lower()
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if m == "detect":
            return "detect"
        if m == "segment":
            return "segment"
        if m == "pose":
            return "pose"
        if m == "obb":
            return "obb"


    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)


    if isinstance(model, nn.Module):
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))

        for m in model.modules():
            if isinstance(m, Segment):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect, WorldDetect)):
                return "detect"


    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"


    LOGGER.warning(
        "WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"
