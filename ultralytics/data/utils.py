import contextlib
import hashlib
import json
import os
import random
import subprocess
import tempfile
import time
import zipfile
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tarfile import is_tarfile

import cv2
import numpy as np
from PIL import Image, ImageOps

from ultralytics.nn.autobackend import check_class_names
from ultralytics.utils import (
    DATASETS_DIR,
    LOGGER,
    NUM_THREADS,
    ROOT,
    SETTINGS_YAML,
    TQDM,
    clean_url,
    colorstr,
    emojis,
    is_dir_writeable,
    yaml_load,
    yaml_save,
)
from ultralytics.utils.checks import check_file, check_font, is_ascii
from ultralytics.utils.downloads import download, safe_download, unzip_file
from ultralytics.utils.ops import segments2boxes

HELP_URL = "See https://docs.ultralytics.com/datasets/detect for dataset formatting guidance."
IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"}
VID_FORMATS = {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"}
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"
FORMATS_HELP_MSG = f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"


def img2label_paths(img_paths):

    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]

def imgir2label_paths(img_paths):

    sa, sb = f"{os.sep}image{os.sep}", f"{os.sep}labels{os.sep}"
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]

def get_hash(paths):

    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
    h = hashlib.sha256(str(size).encode())
    h.update("".join(paths).encode())
    return h.hexdigest()


def exif_size(img: Image.Image):

    s = img.size
    if img.format == "JPEG":
        with contextlib.suppress(Exception):
            exif = img.getexif()
            if exif:
                rotation = exif.get(274, None)
                if rotation in {6, 8}:
                    s = s[1], s[0]
    return s


def verify_image(args):

    (im_file, cls), prefix = args

    nf, nc, msg = 0, 0, ""
    try:
        im = Image.open(im_file)
        im.verify()
        shape = exif_size(im)
        shape = (shape[1], shape[0])
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"Invalid image format {im.format}. {FORMATS_HELP_MSG}"
        if im.format.lower() in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"
        nf = 1
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
    return (im_file, cls), nf, nc, msg


def verify_image_label(args):

    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim = args

    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None
    try:

        im = Image.open(im_file)
        im.verify()
        shape = exif_size(im)
        shape = (shape[1], shape[0])
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}. {FORMATS_HELP_MSG}"
        if im.format.lower() in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"


        if os.path.isfile(lb_file):
            nf = 1
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb) and (not keypoint):
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                if keypoint:
                    assert lb.shape[1] == (5 + nkpt * ndim), f"labels require {(5 + nkpt * ndim)} columns each"
                    points = lb[:, 5:].reshape(-1, ndim)[:, :2]
                else:
                    assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                    points = lb[:, 1:]
                assert points.max() <= 1, f"non-normalized or out of bounds coordinates {points[points > 1]}"
                assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"


                max_cls = lb[:, 0].max()
                assert max_cls <= num_cls, (
                    f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                    f"Possible class labels are 0-{num_cls - 1}"
                )
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:
                    lb = lb[i]
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1
                lb = np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)
        else:
            nm = 1
            lb = np.zeros((0, (5 + nkpt * ndim) if keypoints else 5), dtype=np.float32)
        if keypoint:
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)
        lb = lb[:, :5]
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, None, nm, nf, ne, nc, msg]


def polygon2mask(imgsz, polygons, color=1, downsample_ratio=1):













    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons, dtype=np.int32)
    polygons = polygons.reshape((polygons.shape[0], -1, 2))
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)

    return cv2.resize(mask, (nw, nh))


def polygons2masks(imgsz, polygons, color, downsample_ratio=1):













    return np.array([polygon2mask(imgsz, [x.reshape(-1)], color, downsample_ratio) for x in polygons])


def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):

    masks = np.zeros(
        (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
        dtype=np.int32 if len(segments) > 255 else np.uint8,
    )
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(imgsz, [segments[si].reshape(-1)], downsample_ratio=downsample_ratio, color=1)
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index


def find_dataset_yaml(path: Path) -> Path:













    files = list(path.glob("*.yaml")) or list(path.rglob("*.yaml"))
    assert files, f"No YAML file found in '{path.resolve()}'"
    if len(files) > 1:
        files = [f for f in files if f.stem == path.stem]
    assert len(files) == 1, f"Expected 1 YAML file in '{path.resolve()}', but found {len(files)}.\n{files}"
    return files[0]


def check_det_dataset(dataset, autodownload=True):















    file = check_file(dataset)


    extract_dir = ""
    if zipfile.is_zipfile(file) or is_tarfile(file):
        new_dir = safe_download(file, dir=DATASETS_DIR, unzip=True, delete=False)
        file = find_dataset_yaml(DATASETS_DIR / new_dir)
        extract_dir, autodownload = file.parent, False


    data = yaml_load(file, append_filename=True)


    for k in "train", "val","train_ir","val_ir":
        if k not in data:
            if k != "val_rgb" or "validation" not in data:
                raise SyntaxError(
                    emojis(f"{dataset} '{k}:' key missing ❌.\n'train' and 'val' are required in all data YAMLs.")
                )
            LOGGER.info("WARNING ⚠️ renaming data YAML 'validation' key to 'val' to match YOLO format.")
            data["val_rgb"] = data.pop("validation")
    if "names" not in data and "nc" not in data:
        raise SyntaxError(emojis(f"{dataset} key missing ❌.\n either 'names' or 'nc' are required in all data YAMLs."))
    if "names" in data and "nc" in data and len(data["names"]) != data["nc"]:
        raise SyntaxError(emojis(f"{dataset} 'names' length {len(data['names'])} and 'nc: {data['nc']}' must match."))
    if "names" not in data:
        data["names"] = [f"class_{i}" for i in range(data["nc"])]
    else:
        data["nc"] = len(data["names"])

    data["names"] = check_class_names(data["names"])


    path = Path(extract_dir or data.get("path") or Path(data.get("yaml_file", "")).parent)
    if not path.is_absolute():
        path = (DATASETS_DIR / path).resolve()


    data["path"] = path
    for k in "train", "val","train_ir","val_ir", "test", "test_ir", "minival":
        if data.get(k):
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]


    val, s = (data.get(x) for x in ("val", "download"))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]
        if not all(x.exists() for x in val):
            name = clean_url(dataset)
            m = f"\nDataset '{name}' images not found ⚠️, missing path '{[x for x in val if not x.exists()][0]}'"
            if s and autodownload:
                LOGGER.warning(m)
            else:
                m += f"\nNote dataset download directory is '{DATASETS_DIR}'. You can update this in '{SETTINGS_YAML}'"
                raise FileNotFoundError(m)
            t = time.time()
            r = None
            if s.startswith("http") and s.endswith(".zip"):
                safe_download(url=s, dir=DATASETS_DIR, delete=True)
            elif s.startswith("bash "):
                LOGGER.info(f"Running {s} ...")
                r = os.system(s)
            else:
                exec(s, {"yaml": data})
            dt = f"({round(time.time() - t, 1)}s)"
            s = f"success ✅ {dt}, saved to {colorstr('bold', DATASETS_DIR)}" if r in {0, None} else f"failure {dt} ❌"
            LOGGER.info(f"Dataset download {s}\n")


    return data


def check_cls_dataset(dataset, split=""):




















    if str(dataset).startswith(("http:/", "https:/")):
        dataset = safe_download(dataset, dir=DATASETS_DIR, unzip=True, delete=False)
    elif Path(dataset).suffix in {".zip", ".tar", ".gz"}:
        file = check_file(dataset)
        dataset = safe_download(file, dir=DATASETS_DIR, unzip=True, delete=False)

    dataset = Path(dataset)
    data_dir = (dataset if dataset.is_dir() else (DATASETS_DIR / dataset)).resolve()
    if not data_dir.is_dir():
        LOGGER.warning(f"\nDataset not found ⚠️, missing path {data_dir}, attempting download...")
        t = time.time()
        if str(dataset) == "imagenet":
            subprocess.run(f"bash {ROOT / 'data/scripts/get_imagenet.sh'}", shell=True, check=True)
        else:
            url = f"https://github.com/ultralytics/yolov5/releases/download/v1.0/{dataset}.zip"
            download(url, dir=data_dir.parent)
        s = f"Dataset download success ✅ ({time.time() - t:.1f}s), saved to {colorstr('bold', data_dir)}\n"
        LOGGER.info(s)
    train_set = data_dir / "train"
    val_set = (
        data_dir / "val"
        if (data_dir / "val").exists()
        else data_dir / "validation"
        if (data_dir / "validation").exists()
        else None
    )
    test_set = data_dir / "test" if (data_dir / "test").exists() else None
    if split == "val" and not val_set:
        LOGGER.warning("WARNING ⚠️ Dataset 'split=val' not found, using 'split=test' instead.")
    elif split == "test" and not test_set:
        LOGGER.warning("WARNING ⚠️ Dataset 'split=test' not found, using 'split=val' instead.")

    nc = len([x for x in (data_dir / "train").glob("*") if x.is_dir()])
    names = [x.name for x in (data_dir / "train").iterdir() if x.is_dir()]
    names = dict(enumerate(sorted(names)))


    for k, v in {"train": train_set, "val": val_set, "test": test_set}.items():
        prefix = f'{colorstr(f"{k}:")} {v}...'
        if v is None:
            LOGGER.info(prefix)
        else:
            files = [path for path in v.rglob("*.*") if path.suffix[1:].lower() in IMG_FORMATS]
            nf = len(files)
            nd = len({file.parent for file in files})
            if nf == 0:
                if k == "train":
                    raise FileNotFoundError(emojis(f"{dataset} '{k}:' no training images found ❌ "))
                else:
                    LOGGER.warning(f"{prefix} found {nf} images in {nd} classes: WARNING ⚠️ no images found")
            elif nd != nc:
                LOGGER.warning(f"{prefix} found {nf} images in {nd} classes: ERROR ❌️ requires {nc} classes, not {nd}")
            else:
                LOGGER.info(f"{prefix} found {nf} images in {nd} classes ✅ ")

    return {"train": train_set, "val": val_set, "test": test_set, "nc": nc, "names": names}


class HUBDatasetStats:
























    def __init__(self, path="coco8.yaml", task="detect", autodownload=False):

        path = Path(path).resolve()
        LOGGER.info(f"Starting HUB dataset checks for {path}....")

        self.task = task
        if self.task == "classify":
            unzip_dir = unzip_file(path)
            data = check_cls_dataset(unzip_dir)
            data["path"] = unzip_dir
        else:
            _, data_dir, yaml_path = self._unzip(Path(path))
            try:

                data = yaml_load(yaml_path)
                data["path"] = ""
                yaml_save(yaml_path, data)
                data = check_det_dataset(yaml_path, autodownload)
                data["path"] = data_dir
            except Exception as e:
                raise Exception("error/HUB/dataset_stats/init") from e

        self.hub_dir = Path(f'{data["path"]}-hub')
        self.im_dir = self.hub_dir / "images"
        self.stats = {"nc": len(data["names"]), "names": list(data["names"].values())}
        self.data = data

    @staticmethod
    def _unzip(path):

        if not str(path).endswith(".zip"):
            return False, None, path
        unzip_dir = unzip_file(path, path=path.parent)
        assert unzip_dir.is_dir(), (
            f"Error unzipping {path}, {unzip_dir} not found. " f"path/to/abc.zip MUST unzip to path/to/abc/"
        )
        return True, str(unzip_dir), find_dataset_yaml(unzip_dir)

    def _hub_ops(self, f):

        compress_one_image(f, self.im_dir / Path(f).name)

    def get_json(self, save=False, verbose=False):


        def _round(labels):

            if self.task == "detect":
                coordinates = labels["bboxes"]
            elif self.task == "segment":
                coordinates = [x.flatten() for x in labels["segments"]]
            elif self.task == "pose":
                n = labels["keypoints"].shape[0]
                coordinates = np.concatenate((labels["bboxes"], labels["keypoints"].reshape(n, -1)), 1)
            else:
                raise ValueError("Undefined dataset task.")
            zipped = zip(labels["cls"], coordinates)
            return [[int(c[0]), *(round(float(x), 4) for x in points)] for c, points in zipped]

        for split in "train", "val", "test":
            self.stats[split] = None
            path = self.data.get(split)


            if path is None:
                continue
            files = [f for f in Path(path).rglob("*.*") if f.suffix[1:].lower() in IMG_FORMATS]
            if not files:
                continue


            if self.task == "classify":
                from torchvision.datasets import ImageFolder

                dataset = ImageFolder(self.data[split])

                x = np.zeros(len(dataset.classes)).astype(int)
                for im in dataset.imgs:
                    x[im[1]] += 1

                self.stats[split] = {
                    "instance_stats": {"total": len(dataset), "per_class": x.tolist()},
                    "image_stats": {"total": len(dataset), "unlabelled": 0, "per_class": x.tolist()},
                    "labels": [{Path(k).name: v} for k, v in dataset.imgs],
                }
            else:
                from ultralytics.data import YOLODataset

                dataset = YOLODataset(img_path=self.data[split], data=self.data, task=self.task)
                x = np.array(
                    [
                        np.bincount(label["cls"].astype(int).flatten(), minlength=self.data["nc"])
                        for label in TQDM(dataset.labels, total=len(dataset), desc="Statistics")
                    ]
                )
                self.stats[split] = {
                    "instance_stats": {"total": int(x.sum()), "per_class": x.sum(0).tolist()},
                    "image_stats": {
                        "total": len(dataset),
                        "unlabelled": int(np.all(x == 0, 1).sum()),
                        "per_class": (x > 0).sum(0).tolist(),
                    },
                    "labels": [{Path(k).name: _round(v)} for k, v in zip(dataset.im_files, dataset.labels)],
                }


        if save:
            self.hub_dir.mkdir(parents=True, exist_ok=True)
            stats_path = self.hub_dir / "stats.json"
            LOGGER.info(f"Saving {stats_path.resolve()}...")
            with open(stats_path, "w") as f:
                json.dump(self.stats, f)
        if verbose:
            LOGGER.info(json.dumps(self.stats, indent=2, sort_keys=False))
        return self.stats

    def process_images(self):

        from ultralytics.data import YOLODataset

        self.im_dir.mkdir(parents=True, exist_ok=True)
        for split in "train", "val", "test":
            if self.data.get(split) is None:
                continue
            dataset = YOLODataset(img_path=self.data[split], data=self.data)
            with ThreadPool(NUM_THREADS) as pool:
                for _ in TQDM(pool.imap(self._hub_ops, dataset.im_files), total=len(dataset), desc=f"{split} images"):
                    pass
        LOGGER.info(f"Done. All images saved to {self.im_dir}")
        return self.im_dir


def compress_one_image(f, f_new=None, max_dim=1920, quality=50):





















    try:
        im = Image.open(f)
        r = max_dim / max(im.height, im.width)
        if r < 1.0:
            im = im.resize((int(im.width * r), int(im.height * r)))
        im.save(f_new or f, "JPEG", quality=quality, optimize=True)
    except Exception as e:
        LOGGER.info(f"WARNING ⚠️ HUB ops PIL failure {f}: {e}")
        im = cv2.imread(f)
        im_height, im_width = im.shape[:2]
        r = max_dim / max(im_height, im_width)
        if r < 1.0:
            im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(f_new or f), im)


def autosplit(path=DATASETS_DIR / "coco8/images", weights=(0.9, 0.1, 0.0), annotated_only=False):
















    path = Path(path)
    files = sorted(x for x in path.rglob("*.*") if x.suffix[1:].lower() in IMG_FORMATS)
    n = len(files)
    random.seed(0)
    indices = random.choices([0, 1, 2], weights=weights, k=n)

    txt = ["autosplit_train.txt", "autosplit_val.txt", "autosplit_test.txt"]
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()

    LOGGER.info(f"Autosplitting images from {path}" + ", using *.txt labeled images only" * annotated_only)
    for i, img in TQDM(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():
            with open(path.parent / txt[i], "a") as f:
                f.write(f"./{img.relative_to(path.parent).as_posix()}" + "\n")


def load_dataset_cache_file(path):

    import gc

    gc.disable()
    cache = np.load(str(path), allow_pickle=True).item()
    gc.enable()
    return cache


def save_dataset_cache_file(prefix, path, x, version):

    x["version"] = version
    if is_dir_writeable(path.parent):


        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                np.save(stream, x)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        LOGGER.info(f"{prefix}New cache created: {path}")
    else:
        LOGGER.warning(f"{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.")
