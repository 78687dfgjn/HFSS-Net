import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import dataloader, distributed

from ultralytics.data.loaders import (
    LOADERS,
    LoadImagesAndVideos,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    SourceTypes,
    autocast_list,
)
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.checks import check_file

from .dataset import GroundingDataset, YOLODataset, YOLOMultiModalDataset
from .utils import PIN_MEMORY


class InfiniteDataLoader(dataloader.DataLoader):






    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):

        return len(self.batch_sampler.sampler)

    def __iter__(self):

        for _ in range(len(self)):
            yield next(self.iterator)

    def reset(self):





        self.iterator = self._get_iterator()


class _RepeatSampler:







    def __init__(self, sampler):

        self.sampler = sampler

    def __iter__(self):

        while True:
            yield from iter(self.sampler)


def seed_worker(worker_id):

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    audit_dir = getattr(torch.utils.data.get_worker_info().dataset, "_seed_audit_dir", None)
    if audit_dir:
        import json
        Path(audit_dir).mkdir(parents=True, exist_ok=True)
        audit_path = Path(audit_dir) / f"worker_{worker_id}_{os.getpid()}.json"
        audit_tmp = audit_path.with_suffix('.tmp')
        audit_tmp.write_text(json.dumps({
            "worker_id": worker_id, "torch_initial_seed": torch.initial_seed(),
            "numpy_python_seed": worker_seed,
        }))
        audit_tmp.replace(audit_path)


def build_yolo_dataset(cfg, img_path, imgir_path,batch, data, mode="train", rect=False, stride=32, multi_modal=False):

    dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
    return dataset(
        img_path=img_path,
        imgir_path=imgir_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def build_grounding(cfg, img_path, json_file, batch, mode="train", rect=False, stride=32):

    return GroundingDataset(
        img_path=img_path,
        json_file=json_file,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1, seed=0):

    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()
    nw = min([os.cpu_count() // max(nd, 1), workers])
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle, seed=int(seed))
    generator = torch.Generator()



    generator.manual_seed(6148914691236517205 + RANK + int(seed))
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def check_source(source):

    webcam, screenshot, from_img, in_memory, tensor = False, False, False, False, False
    if isinstance(source, (str, int, Path)):
        source = str(source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS | VID_FORMATS)
        is_url = source.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://"))
        webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
        screenshot = source.lower() == "screen"
        if is_url and is_file:
            source = check_file(source)
    elif isinstance(source, LOADERS):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source)
        from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError("Unsupported image type. For supported types see https://docs.ultralytics.com/modes/predict")

    return source, webcam, screenshot, from_img, in_memory, tensor


def load_inference_source(source=None, batch=1, vid_stride=1, buffer=False):












    source, stream, screenshot, from_img, in_memory, tensor = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(stream, screenshot, from_img, tensor)


    if tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif stream:
        dataset = LoadStreams(source, vid_stride=vid_stride, buffer=buffer)
    elif screenshot:
        dataset = LoadScreenshots(source)
    elif from_img:
        dataset = LoadPilAndNumpy(source)
    else:
        dataset = LoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride)


    setattr(dataset, "source_type", source_type)

    return dataset
