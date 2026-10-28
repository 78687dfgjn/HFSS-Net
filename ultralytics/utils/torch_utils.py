import math
import os
import random
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import (
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_KEYS,
    LOGGER,
    PYTHON_VERSION,
    TORCHVISION_VERSION,
    __version__,
    colorstr,
)
from ultralytics.utils.checks import check_version

try:
    import thop
except ImportError:
    thop = None


TORCH_1_9 = check_version(torch.__version__, "1.9.0")
TORCH_1_13 = check_version(torch.__version__, "1.13.0")
TORCH_2_0 = check_version(torch.__version__, "2.0.0")
TORCHVISION_0_10 = check_version(TORCHVISION_VERSION, "0.10.0")
TORCHVISION_0_11 = check_version(TORCHVISION_VERSION, "0.11.0")
TORCHVISION_0_13 = check_version(TORCHVISION_VERSION, "0.13.0")


@contextmanager
def torch_distributed_zero_first(local_rank: int):

    initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
    if initialized and local_rank not in {-1, 0}:
        dist.barrier(device_ids=[local_rank])
    yield
    if initialized and local_rank == 0:
        dist.barrier(device_ids=[0])


def smart_inference_mode():


    def decorate(fn):

        if TORCH_1_9 and torch.is_inference_mode_enabled():
            return fn
        else:
            return (torch.inference_mode if TORCH_1_9 else torch.no_grad)()(fn)

    return decorate


def get_cpu_info():

    import cpuinfo

    k = "brand_raw", "hardware_raw", "arch_string_raw"
    info = cpuinfo.get_cpu_info()
    string = info.get(k[0] if k[0] in info else k[1] if k[1] in info else k[2], "unknown")
    return string.replace("(R)", "").replace("CPU ", "").replace("@ ", "")


def select_device(device="", batch=0, newline=False, verbose=True):

































    if isinstance(device, torch.device):
        return device

    s = f"Ultralytics YOLOv{__version__} 🚀 Python-{PYTHON_VERSION} torch-{torch.__version__} "
    device = str(device).lower()
    for remove in "cuda:", "none", "(", ")", "[", "]", "'", " ":
        device = device.replace(remove, "")
    cpu = device == "cpu"
    mps = device in {"mps", "mps:0"}
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device:
        if device == "cuda":
            device = "0"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if os.environ.get('SLURM_JOB_ID') and visible:
            allocated = os.environ.setdefault('PTANET_SLURM_CUDA_DEVICES', visible).split(',')
            requested = [int(i) for i in device.split(',')]
            if any(i < 0 or i >= len(allocated) for i in requested):
                raise ValueError(f'Logical devices {device} exceed Slurm allocation {allocated}')
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(allocated[i] for i in requested)
            LOGGER.info(f'Slurm allocated CUDA devices: {allocated}; selected physical IDs: '
                        f'{os.environ["CUDA_VISIBLE_DEVICES"]}')
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = device
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= len(device.split(","))):
            LOGGER.info(s)
            install = (
                "See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no "
                "CUDA devices are seen by torch.\n"
                if torch.cuda.device_count() == 0
                else ""
            )
            raise ValueError(
                f"Invalid CUDA 'device={device}' requested."
                f" Use 'device=cpu' or pass valid CUDA device(s) if available,"
                f" i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n"
                f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}"
                f"\ntorch.cuda.device_count(): {torch.cuda.device_count()}"
                f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n"
                f"{install}"
            )

    if not cpu and not mps and torch.cuda.is_available():
        devices = device.split(",") if device else "0"
        n = len(devices)
        if n > 1 and batch > 0 and batch % n != 0:
            raise ValueError(
                f"'batch={batch}' must be a multiple of GPU count {n}. Try 'batch={batch // n * n}' or "
                f"'batch={batch // n * n + n}', the nearest batch sizes evenly divisible by {n}."
            )
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"
        arg = "cuda:0"
    elif mps and TORCH_2_0 and torch.backends.mps.is_available():

        s += f"MPS ({get_cpu_info()})\n"
        arg = "mps"
    else:
        s += f"CPU ({get_cpu_info()})\n"
        arg = "cpu"

    if verbose:
        LOGGER.info(s if newline else s.rstrip())
    return torch.device(arg)


def time_sync():

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def fuse_conv_and_bn(conv, bn):

    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )


    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))


    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def fuse_deconv_and_bn(deconv, bn):

    fuseddconv = (
        nn.ConvTranspose2d(
            deconv.in_channels,
            deconv.out_channels,
            kernel_size=deconv.kernel_size,
            stride=deconv.stride,
            padding=deconv.padding,
            output_padding=deconv.output_padding,
            dilation=deconv.dilation,
            groups=deconv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(deconv.weight.device)
    )


    w_deconv = deconv.weight.clone().view(deconv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fuseddconv.weight.copy_(torch.mm(w_bn, w_deconv).view(fuseddconv.weight.shape))


    b_conv = torch.zeros(deconv.weight.shape[1], device=deconv.weight.device) if deconv.bias is None else deconv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fuseddconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fuseddconv


def get_flops(model, imgsz=640):

    if not thop:
        return 0.0

    try:
        model = de_parallel(model)
        p = next(model.parameters())
        if not isinstance(imgsz, list):
            imgsz = [imgsz, imgsz]
        try:

            stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32

            im = torch.zeros((1, 6, stride, stride), device=p.device)
            flops = thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2
            return flops * imgsz[0] / stride * imgsz[1] / stride
        except Exception:

            im = torch.empty((1, p.shape[1], *imgsz), device=p.device)
            return thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2
    except Exception:
        return 0.0

def model_info(model, detailed=False, verbose=True, imgsz=640):





    if not verbose:
        return
    n_p = get_num_params(model)
    n_g = get_num_gradients(model)
    n_l = len(list(model.modules()))
    if detailed:
        LOGGER.info(
            f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}"
        )
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            LOGGER.info(
                "%5g %40s %9s %12g %20s %10.3g %10.3g %10s"
                % (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std(), p.dtype)
            )

    flops = get_flops(model, imgsz)
    fused = " (fused)" if getattr(model, "is_fused", lambda: False)() else ""
    fs = f", {flops:.1f} GFLOPs" if flops else ""
    yaml_file = getattr(model, "yaml_file", "") or getattr(model, "yaml", {}).get("yaml_file", "")
    model_name = Path(yaml_file).stem.replace("yolo", "YOLO") or "Model"
    LOGGER.info(f"{model_name} summary{fused}: {n_l} layers, {n_p} parameters, {n_g} gradients{fs}")
    return n_l, n_p, n_g, flops


def get_num_params(model):

    return sum(x.numel() for x in model.parameters())


def get_num_gradients(model):

    return sum(x.numel() for x in model.parameters() if x.requires_grad)


def model_info_for_loggers(trainer):













    if trainer.args.profile:
        from ultralytics.utils.benchmarks import ProfileModels

        results = ProfileModels([trainer.last], device=trainer.device).profile()[0]
        results.pop("model/name")
    else:
        results = {
            "model/parameters": get_num_params(trainer.model),
            "model/GFLOPs": round(get_flops(trainer.model), 3),
        }
    results["model/speed_PyTorch(ms)"] = round(trainer.validator.speed["inference"], 3)
    return results





def get_flops_with_torch_profiler(model, imgsz=640):

    if TORCH_2_0:
        model = de_parallel(model)
        p = next(model.parameters())
        stride = (max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32) * 2
        im = torch.zeros((1, p.shape[1], stride, stride), device=p.device)
        with torch.profiler.profile(with_flops=True) as prof:
            model(im)
        flops = sum(x.flops for x in prof.key_averages()) / 1e9
        imgsz = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]
        flops = flops * imgsz[0] / stride * imgsz[1] / stride
        return flops
    return 0


def initialize_weights(model):

    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            m.inplace = True


def scale_img(img, ratio=1.0, same_shape=False, gs=32):



    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)
    if not same_shape:
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)


def make_divisible(x, divisor):

    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor


def copy_attr(a, b, include=(), exclude=()):

    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


def get_latest_opset():


    return max(int(k[14:]) for k in vars(torch.onnx) if "symbolic_opset" in k) - 1


def intersect_dicts(da, db, exclude=()):

    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


def is_parallel(model):

    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def de_parallel(model):

    return model.module if is_parallel(model) else model


def one_cycle(y1=0.0, y2=1.0, steps=100):

    return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1


def init_seeds(seed=0, deterministic=False):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        if TORCH_2_0:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            os.environ["PYTHONHASHSEED"] = str(seed)
        else:
            LOGGER.warning("WARNING ⚠️ Upgrade to torch>=2.0.0 for deterministic training.")
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False


class ModelEMA:






    def __init__(self, model, decay=0.9999, tau=2000, updates=0):

        self.ema = deepcopy(de_parallel(model)).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):

        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()


    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):

        if self.enabled:
            copy_attr(self.ema, model, include, exclude)


def strip_optimizer(f: Union[str, Path] = "best.pt", s: str = "") -> None:



















    x = torch.load(f, map_location=torch.device("cpu"))
    if "model" not in x:
        LOGGER.info(f"Skipping {f}, not a valid Ultralytics model.")
        return

    if hasattr(x["model"], "args"):
        x["model"].args = dict(x["model"].args)
    args = {**DEFAULT_CFG_DICT, **x["train_args"]} if "train_args" in x else None
    if x.get("ema"):
        x["model"] = x["ema"]
    for k in "optimizer", "best_fitness", "ema", "updates":
        x[k] = None
    x["epoch"] = -1
    x["model"].half()
    for p in x["model"].parameters():
        p.requires_grad = False
    x["train_args"] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}

    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1e6
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")


def convert_optimizer_state_dict_to_fp16(state_dict):





    for state in state_dict["state"].values():
        for k, v in state.items():
            if k != "step" and isinstance(v, torch.Tensor) and v.dtype is torch.float32:
                state[k] = v.half()

    return state_dict


def profile(input, ops, n=10, device=None):













    results = []
    if not isinstance(device, torch.device):
        device = select_device(device)
    LOGGER.info(
        f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
        f"{'input':>24s}{'output':>24s}"
    )

    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)
        x.requires_grad = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, "to") else m
            m = m.half() if hasattr(m, "half") and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            tf, tb, t = 0, 0, [0, 0, 0]
            try:
                flops = thop.profile(m, inputs=[x], verbose=False)[0] / 1e9 * 2 if thop else 0
            except Exception:
                flops = 0

            try:
                for _ in range(n):
                    t[0] = time_sync()
                    y = m(x)
                    t[1] = time_sync()
                    try:
                        (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                        t[2] = time_sync()
                    except Exception:

                        t[2] = float("nan")
                    tf += (t[1] - t[0]) * 1000 / n
                    tb += (t[2] - t[1]) * 1000 / n
                mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
                s_in, s_out = (tuple(x.shape) if isinstance(x, torch.Tensor) else "list" for x in (x, y))
                p = sum(x.numel() for x in m.parameters()) if isinstance(m, nn.Module) else 0
                LOGGER.info(f"{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}")
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                LOGGER.info(e)
                results.append(None)
            torch.cuda.empty_cache()
    return results


class EarlyStopping:


    def __init__(self, patience=50):






        self.best_fitness = 0.0
        self.best_epoch = 0
        self.patience = patience or float("inf")
        self.possible_stop = False

    def __call__(self, epoch, fitness):










        if fitness is None:
            return False

        if fitness >= self.best_fitness:
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch
        self.possible_stop = delta >= (self.patience - 1)
        stop = delta >= self.patience
        if stop:
            prefix = colorstr("EarlyStopping: ")
            LOGGER.info(
                f"{prefix}Training stopped early as no improvement observed in last {self.patience} epochs. "
                f"Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n"
                f"To update EarlyStopping(patience={self.patience}) pass a new patience value, "
                f"i.e. `patience=300` or use `patience=0` to disable EarlyStopping."
            )
        return stop
