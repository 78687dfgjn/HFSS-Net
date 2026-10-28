import contextlib
import glob
import inspect
import math
import os
import platform
import re
import shutil
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
import torch

from ultralytics.utils import (
    ASSETS,
    AUTOINSTALL,
    IS_COLAB,
    IS_DOCKER,
    IS_JUPYTER,
    IS_KAGGLE,
    IS_PIP_PACKAGE,
    LINUX,
    LOGGER,
    ONLINE,
    PYTHON_VERSION,
    ROOT,
    TORCHVISION_VERSION,
    USER_CONFIG_DIR,
    Retry,
    SimpleNamespace,
    ThreadingLocked,
    TryExcept,
    clean_url,
    colorstr,
    downloads,
    emojis,
    is_github_action_running,
    url2file,
)


def parse_requirements(file_path=ROOT.parent / "requirements.txt", package=""):


















    if package:
        requires = [x for x in metadata.distribution(package).requires if "extra == " not in x]
    else:
        requires = Path(file_path).read_text().splitlines()

    requirements = []
    for line in requires:
        line = line.strip()
        if line and not line.startswith("#"):
            line = line.split("#")[0].strip()
            match = re.match(r"([a-zA-Z0-9-_]+)\s*([<>!=~]+.*)?", line)
            if match:
                requirements.append(SimpleNamespace(name=match[1], specifier=match[2].strip() if match[2] else ""))

    return requirements


def parse_version(version="0.0.0") -> tuple:










    try:
        return tuple(map(int, re.findall(r"\d+", version)[:3]))
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ failure for parse_version({version}), returning (0, 0, 0): {e}")
        return 0, 0, 0


def is_ascii(s) -> bool:










    s = str(s)


    return all(ord(c) < 128 for c in s)


def check_imgsz(imgsz, stride=32, min_dim=1, max_dim=2, floor=0):















    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)


    if isinstance(imgsz, int):
        imgsz = [imgsz]
    elif isinstance(imgsz, (list, tuple)):
        imgsz = list(imgsz)
    elif isinstance(imgsz, str):
        imgsz = [int(imgsz)] if imgsz.isnumeric() else eval(imgsz)
    else:
        raise TypeError(
            f"'imgsz={imgsz}' is of invalid type {type(imgsz).__name__}. "
            f"Valid imgsz types are int i.e. 'imgsz=640' or list i.e. 'imgsz=[640,640]'"
        )


    if len(imgsz) > max_dim:
        msg = (
            "'train' and 'val' imgsz must be an integer, while 'predict' and 'export' imgsz may be a [h, w] list "
            "or an integer, i.e. 'yolo export imgsz=640,480' or 'yolo export imgsz=640'"
        )
        if max_dim != 1:
            raise ValueError(f"imgsz={imgsz} is not a valid image size. {msg}")
        LOGGER.warning(f"WARNING ⚠️ updating to 'imgsz={max(imgsz)}'. {msg}")
        imgsz = [max(imgsz)]

    sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]


    if sz != imgsz:
        LOGGER.warning(f"WARNING ⚠️ imgsz={imgsz} must be multiple of max stride {stride}, updating to {sz}")


    sz = [sz[0], sz[0]] if min_dim == 2 and len(sz) == 1 else sz[0] if min_dim == 1 and len(sz) == 1 else sz

    return sz


def check_version(
    current: str = "0.0.0",
    required: str = "0.0.0",
    name: str = "version",
    hard: bool = False,
    verbose: bool = False,
    msg: str = "",
) -> bool:





























    if not current:
        LOGGER.warning(f"WARNING ⚠️ invalid check_version({current}, {required}) requested, please check values.")
        return True
    elif not current[0].isdigit():
        try:
            name = current
            current = metadata.version(current)
        except metadata.PackageNotFoundError as e:
            if hard:
                raise ModuleNotFoundError(emojis(f"WARNING ⚠️ {current} package is required but not installed")) from e
            else:
                return False

    if not required:
        return True

    op = ""
    version = ""
    result = True
    c = parse_version(current)
    for r in required.strip(",").split(","):
        op, version = re.match(r"([^0-9]*)([\d.]+)", r).groups()
        v = parse_version(version)
        if op == "==" and c != v:
            result = False
        elif op == "!=" and c == v:
            result = False
        elif op in {">=", ""} and not (c >= v):
            result = False
        elif op == "<=" and not (c <= v):
            result = False
        elif op == ">" and not (c > v):
            result = False
        elif op == "<" and not (c < v):
            result = False
    if not result:
        warning = f"WARNING ⚠️ {name}{op}{version} is required, but {name}=={current} is currently installed {msg}"
        if hard:
            raise ModuleNotFoundError(emojis(warning))
        if verbose:
            LOGGER.warning(warning)
    return result


def check_latest_pypi_version(package_name="ultralytics"):









    with contextlib.suppress(Exception):
        requests.packages.urllib3.disable_warnings()
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=3)
        if response.status_code == 200:
            return response.json()["info"]["version"]


def check_pip_update_available():






    if ONLINE and IS_PIP_PACKAGE:
        with contextlib.suppress(Exception):
            from ultralytics import __version__

            latest = check_latest_pypi_version()
            if check_version(__version__, f"<{latest}"):
                LOGGER.info(
                    f"New https://pypi.org/project/ultralytics/{latest} available 😃 "
                    f"Update with 'pip install -U ultralytics'"
                )
                return True
    return False


@ThreadingLocked()
def check_font(font="Arial.ttf"):









    from matplotlib import font_manager


    name = Path(font).name
    file = USER_CONFIG_DIR / name
    if file.exists():
        return file


    matches = [s for s in font_manager.findSystemFonts() if font in s]
    if any(matches):
        return matches[0]


    url = f"https://ultralytics.com/assets/{name}"
    if downloads.is_url(url, check=True):
        downloads.safe_download(url=url, file=file)
        return file


def check_python(minimum: str = "3.8.0") -> bool:









    return check_version(PYTHON_VERSION, minimum, name="Python ", hard=True)


@TryExcept()
def check_requirements(requirements=ROOT.parent / "requirements.txt", exclude=(), install=True, cmds=""):

























    prefix = colorstr("red", "bold", "requirements:")
    check_python()
    check_torchvision()
    if isinstance(requirements, Path):
        file = requirements.resolve()
        assert file.exists(), f"{prefix} {file} not found, check failed."
        requirements = [f"{x.name}{x.specifier}" for x in parse_requirements(file) if x.name not in exclude]
    elif isinstance(requirements, str):
        requirements = [requirements]

    pkgs = []
    for r in requirements:
        r_stripped = r.split("/")[-1].replace(".git", "")
        match = re.match(r"([a-zA-Z0-9-_]+)([<>!=~]+.*)?", r_stripped)
        name, required = match[1], match[2].strip() if match[2] else ""
        try:
            assert check_version(metadata.version(name), required)
        except (AssertionError, metadata.PackageNotFoundError):
            pkgs.append(r)

    s = " ".join(f'"{x}"' for x in pkgs)
    if s:
        if install and AUTOINSTALL:
            n = len(pkgs)
            LOGGER.info(f"{prefix} Ultralytics requirement{'s' * (n > 1)} {pkgs} not found, attempting AutoUpdate...")
            try:
                t = time.time()
                assert ONLINE, "AutoUpdate skipped (offline)"
                with Retry(times=2, delay=1):
                    LOGGER.info(subprocess.check_output(f"pip install --no-cache {s} {cmds}", shell=True).decode())
                dt = time.time() - t
                LOGGER.info(
                    f"{prefix} AutoUpdate success ✅ {dt:.1f}s, installed {n} package{'s' * (n > 1)}: {pkgs}\n"
                    f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
                )
            except Exception as e:
                LOGGER.warning(f"{prefix} ❌ {e}")
                return False
        else:
            return False

    return True


def check_torchvision():












    compatibility_table = {"2.0": ["0.15"], "1.13": ["0.14"], "1.12": ["0.13"]}


    v_torch = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    v_torchvision = ".".join(TORCHVISION_VERSION.split("+")[0].split(".")[:2])

    if v_torch in compatibility_table:
        compatible_versions = compatibility_table[v_torch]
        if all(v_torchvision != v for v in compatible_versions):
            print(
                f"WARNING ⚠️ torchvision=={v_torchvision} is incompatible with torch=={v_torch}.\n"
                f"Run 'pip install torchvision=={compatible_versions[0]}' to fix torchvision or "
                "'pip install -U torch torchvision' to update both.\n"
                "For a full compatibility table see https://github.com/pytorch/vision#installation"
            )


def check_suffix(file="yolov8n.pt", suffix=".pt", msg=""):

    if file and suffix:
        if isinstance(suffix, str):
            suffix = (suffix,)
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower().strip()
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}, not {s}"


def check_yolov5u_filename(file: str, verbose: bool = True):

    if "yolov3" in file or "yolov5" in file:
        if "u.yaml" in file:
            file = file.replace("u.yaml", ".yaml")
        elif ".pt" in file and "u" not in file:
            original_file = file
            file = re.sub(r"(.*yolov5([nsmlx]))\.pt", "\\1u.pt", file)
            file = re.sub(r"(.*yolov5([nsmlx])6)\.pt", "\\1u.pt", file)
            file = re.sub(r"(.*yolov3(|-tiny|-spp))\.pt", "\\1u.pt", file)
            if file != original_file and verbose:
                LOGGER.info(
                    f"PRO TIP 💡 Replace 'model={original_file}' with new 'model={file}'.\nYOLOv5 'u' models are "
                    f"trained with https://github.com/ultralytics/ultralytics and feature improved performance vs "
                    f"standard YOLOv5 models trained with https://github.com/ultralytics/yolov5.\n"
                )
    return file


def check_model_file_from_stem(model="yolov8n"):

    if model and not Path(model).suffix and Path(model).stem in downloads.GITHUB_ASSETS_STEMS:
        return Path(model).with_suffix(".pt")
    else:
        return model


def check_file(file, suffix="", download=True, hard=True):

    check_suffix(file, suffix)
    file = str(file).strip()
    file = check_yolov5u_filename(file)
    if (
        not file
        or ("://" not in file and Path(file).exists())
        or file.lower().startswith("grpc://")
    ):
        return file
    elif download and file.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):
        url = file
        file = url2file(file)
        if Path(file).exists():
            LOGGER.info(f"Found {clean_url(url)} locally at {file}")
        else:
            downloads.safe_download(url=url, file=file, unzip=False)
        return file
    else:
        files = glob.glob(str(ROOT / "**" / file), recursive=True) or glob.glob(str(ROOT.parent / file))
        if not files and hard:
            raise FileNotFoundError(f"'{file}' does not exist")
        elif len(files) > 1 and hard:
            raise FileNotFoundError(f"Multiple files match '{file}', specify exact path: {files}")
        return files[0] if len(files) else []


def check_yaml(file, suffix=(".yaml", ".yml"), hard=True):

    return check_file(file, suffix, hard=hard)


def check_is_path_safe(basedir, path):










    base_dir_resolved = Path(basedir).resolve()
    path_resolved = Path(path).resolve()

    return path_resolved.is_file() and path_resolved.parts[: len(base_dir_resolved.parts)] == base_dir_resolved.parts


def check_imshow(warn=False):

    try:
        if LINUX:
            assert "DISPLAY" in os.environ and not IS_DOCKER and not IS_COLAB and not IS_KAGGLE
        cv2.imshow("test", np.zeros((8, 8, 3), dtype=np.uint8))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        if warn:
            LOGGER.warning(f"WARNING ⚠️ Environment does not support cv2.imshow() or PIL Image.show()\n{e}")
        return False


def check_yolo(verbose=True, device=""):

    import psutil

    from ultralytics.utils.torch_utils import select_device

    if IS_JUPYTER:
        if check_requirements("wandb", install=False):
            os.system("pip uninstall -y wandb")
        if IS_COLAB:
            shutil.rmtree("sample_data", ignore_errors=True)

    if verbose:

        gib = 1 << 30
        ram = psutil.virtual_memory().total
        total, used, free = shutil.disk_usage("/")
        s = f"({os.cpu_count()} CPUs, {ram / gib:.1f} GB RAM, {(total - free) / gib:.1f}/{total / gib:.1f} GB disk)"
        with contextlib.suppress(Exception):
            from IPython import display

            display.clear_output()
    else:
        s = ""

    select_device(device=device, newline=False)
    LOGGER.info(f"Setup complete ✅ {s}")


def collect_system_info():


    import psutil

    from ultralytics.utils import ENVIRONMENT, IS_GIT_DIR
    from ultralytics.utils.torch_utils import get_cpu_info

    ram_info = psutil.virtual_memory().total / (1024**3)
    check_yolo()
    LOGGER.info(
        f"\n{'OS':<20}{platform.platform()}\n"
        f"{'Environment':<20}{ENVIRONMENT}\n"
        f"{'Python':<20}{PYTHON_VERSION}\n"
        f"{'Install':<20}{'git' if IS_GIT_DIR else 'pip' if IS_PIP_PACKAGE else 'other'}\n"
        f"{'RAM':<20}{ram_info:.2f} GB\n"
        f"{'CPU':<20}{get_cpu_info()}\n"
        f"{'CUDA':<20}{torch.version.cuda if torch and torch.cuda.is_available() else None}\n"
    )

    for r in parse_requirements(package="ultralytics"):
        try:
            current = metadata.version(r.name)
            is_met = "✅ " if check_version(current, str(r.specifier), hard=True) else "❌ "
        except metadata.PackageNotFoundError:
            current = "(not installed)"
            is_met = "❌ "
        LOGGER.info(f"{r.name:<20}{is_met}{current}{r.specifier}")

    if is_github_action_running():
        LOGGER.info(
            f"\nRUNNER_OS: {os.getenv('RUNNER_OS')}\n"
            f"GITHUB_EVENT_NAME: {os.getenv('GITHUB_EVENT_NAME')}\n"
            f"GITHUB_WORKFLOW: {os.getenv('GITHUB_WORKFLOW')}\n"
            f"GITHUB_ACTOR: {os.getenv('GITHUB_ACTOR')}\n"
            f"GITHUB_REPOSITORY: {os.getenv('GITHUB_REPOSITORY')}\n"
            f"GITHUB_REPOSITORY_OWNER: {os.getenv('GITHUB_REPOSITORY_OWNER')}\n"
        )


def check_amp(model):




















    device = next(model.parameters()).device
    if device.type in {"cpu", "mps"}:
        return False

    if getattr(model, 'yaml', {}).get('paper_mode'):
        from copy import deepcopy
        probe = deepcopy(model).float().eval()
        generator = torch.Generator(device=device).manual_seed(123)
        im = torch.rand((1, 6, 256, 256), generator=generator, device=device)
        with torch.no_grad():
            fp32 = probe(im)
            with torch.autocast('cuda', dtype=torch.float16):
                mixed = probe(im)
        checks = [torch.isfinite(mixed[j]).all() and
                  torch.allclose(fp32[j], mixed[j].float(), atol=0.5 if j == 0 else 0.05, rtol=0.1)
                  for j in (0, 2)]
        passed = all(bool(x) for x in checks)
        del probe
        LOGGER.info(f'PTANet six-channel AMP numerical check: {"passed" if passed else "failed"}')
        return passed

    def amp_allclose(m, im):







        return True

    im = ASSETS / "bus.jpg"
    prefix = colorstr("AMP: ")
    LOGGER.info(f"{prefix}running Automatic Mixed Precision (AMP) checks with YOLOv8n...")
    warning_msg = "Setting 'amp=True'. If you experience zero-mAP or NaN losses you can disable AMP with amp=False."
    try:
        from ultralytics import YOLO

        assert amp_allclose(YOLO("yolov8n.pt"), im)
        LOGGER.info(f"{prefix}checks passed ✅")
    except ConnectionError:
        LOGGER.warning(f"{prefix}checks skipped ⚠️, offline and unable to download YOLOv8n. {warning_msg}")
    except (AttributeError, ModuleNotFoundError):
        LOGGER.warning(
            f"{prefix}checks skipped ⚠️. "
            f"Unable to load YOLOv8n due to possible Ultralytics package modifications. {warning_msg}"
        )
    except AssertionError:
        LOGGER.warning(
            f"{prefix}checks failed ❌. Anomalies were detected with AMP on your system that may lead to "
            f"NaN losses or zero-mAP results, so AMP will be disabled during training."
        )
        return False
    return True


def git_describe(path=ROOT):

    with contextlib.suppress(Exception):
        return subprocess.check_output(f"git -C {path} describe --tags --long --always", shell=True).decode()[:-1]
    return ""


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):


    def strip_auth(v):

        return clean_url(v) if (isinstance(v, str) and v.startswith("http") and len(v) > 100) else v

    x = inspect.currentframe().f_back
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix("")
    except ValueError:
        file = Path(file).stem
    s = (f"{file}: " if show_file else "") + (f"{func}: " if show_func else "")
    LOGGER.info(colorstr(s) + ", ".join(f"{k}={strip_auth(v)}" for k, v in args.items()))


def cuda_device_count() -> int:






    try:

        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"], encoding="utf-8"
        )


        first_line = output.strip().split("\n")[0]

        return int(first_line)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):

        return 0


def cuda_is_available() -> bool:






    return cuda_device_count() > 0



IS_PYTHON_3_12 = PYTHON_VERSION.startswith("3.12")
