import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import psutil
from torch.utils.data import Dataset

from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM

from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS

from ultralytics.utils.ops import xywhn2xyxy
from PIL import  Image
from ultralytics.data.augment import seg_letterbox


class BaseDataset(Dataset):



























    def __init__(
        self,
        img_path,
        imgir_path,
        imgsz=640,
        cache=False,
        augment=False,
        hyp=DEFAULT_CFG,
        prefix="",
        rect=False,
        batch_size=16,
        stride=32,
        pad=0.5,
        single_cls=False,
        classes=None,
        fraction=1.0,
    ):

        super().__init__()
        self.img_path = img_path
        self.imgir_path=imgir_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.im_files = self.get_img_files(self.img_path)
        self.imir_files = self.get_img_files(self.imgir_path)

        self.labels = self.get_labels()

        self.update_labels(include_class=classes)

        self.ni = len(self.labels)
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()


        self.buffer = []
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0


        self.ims,self.imsir, self.im_hw0, self.im_hw = [None] * self.ni,[None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]


        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        if (self.cache == "ram" and self.check_cache_ram()) or self.cache == "disk":
            self.cache_images()



        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path):

        try:
            f = []
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)
                if p.is_dir():
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)

                elif p.is_file():
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]

                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)

            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
        except Exception as e:

            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]
        return im_files




    def update_labels(self, include_class: Optional[list]):

        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def load_image(self, i, rect_mode=True):

        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]


        f1 = f.replace('images', 'image')
        imir=cv2.imread(f1)






        if im is None:
            if fn.exists():
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = cv2.imread(f)
            else:
                im = cv2.imread(f)
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")








































            im = np.dstack((im, imir))
            h0, w0 = im.shape[:2]
            if rect_mode:
                r = self.imgsz / max(h0, w0)
                if r != 1:
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)


            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
                self.buffer.append(i)
                if len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def loadir_image(self, i, rect_mode=True):

        im, f, fn = self.imsir[i], self.imir_files[i], self.npyir_files[i]
        if im is None:
            if fn.exists():
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = cv2.imread(f,cv2.IMREAD_GRAYSCALE)
            else:
                im = cv2.imread(f,cv2.IMREAD_GRAYSCALE)
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")
            h0, w0 = im.shape[:2]
            if rect_mode:
                r = self.imgsz / max(h0, w0)
                if r != 1:
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)


            if self.augment:
                self.imsir[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
                self.buffer.append(i)
                if len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.imsir[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.imsir[i], self.im_hw0[i], self.im_hw[i]

    def generate_segmask(self, i):


        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        im = cv2.imread(f)
        imir = cv2.imread(f.replace('iamges', 'image'))
        im, ratio, pad = seg_letterbox(im, auto=False, Scaleup=self.augment)
        imir, _, _ = seg_letterbox(imir, auto=False, Scaleup=self.augment)

        assert im.shape == imir.shape
        mask = np.zeros_like(im)[:, :, 0]
        mask_h, mask_w = np.shape(mask)

        if self.ni:
            labels = self.labels
            print(type(labels))
            labels = np.array(labels)
            print(labels.shape)
            labels_per_img = xywhn2xyxy(labels[:, 1:], mask_w, mask_h)
            for boxp in labels_per_img:
                mask[int(boxp[1]):int(boxp[3]) + 1, int(boxp[0]):int(boxp[2]) + 1] = 255
        pil_ds = lambda inputs, dsr: np.expand_dims(
            np.array(
                Image.fromarray(inputs).resize((mask_w // dsr, mask_h // dsr), Image.NEAREST),
                inputs.dtype), 0)
        mask8 = pil_ds(mask, 8)
        mask16 = pil_ds(mask, 16)
        mask32 = pil_ds(mask, 32)





        mask8 = np.ascontiguousarray(mask8)
        mask16 = np.ascontiguousarray(mask16)
        mask32 = np.ascontiguousarray(mask32)

        return mask8, mask16, mask32


    def cache_images(self):

        b, gb = 0, 1 << 30
        fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image, "RAM")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
            pbar.close()

    def cacheir_images(self):

        b, gb = 0, 1 << 30

        fcn, storage = (self.cacheir_images_to_disk, "Disk") if self.cache == "disk" else (self.loadir_image, "RAM")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npyir_files[i].stat().st_size
                else:
                    self.imsir[i], self.im_hw0[i], self.im_hw[i] = x
                    b += self.imsir[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
            pbar.close()

    def cache_images(self):

        b, gb = 0, 1 << 30
        fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image, "RAM")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
            pbar.close()
    def cache_images_to_disk(self, i):

        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]), allow_pickle=False)
    def cacheir_images_to_disk(self, i):

        f = self.npyir_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.imir_files[i]), allow_pickle=False)

    def check_cache_ram(self, safety_margin=0.5):

        b, gb = 0, 1 << 30
        n = min(self.ni, 30)
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))
            ratio = self.imgsz / max(im.shape[0], im.shape[1])
            b += im.nbytes * ratio**2
        mem_required = b * self.ni / n * (1 + safety_margin)
        mem = psutil.virtual_memory()
        success = mem_required < mem.available
        if not success:
            self.cache = None
            LOGGER.info(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images ⚠️"
            )
        return success

    def set_rectangle(self):

        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)
        nb = bi[-1] + 1

        s = np.array([x.pop("shape") for x in self.labels])
        ar = s[:, 0] / s[:, 1]
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.imir_files = [self.im_files[i].replace('images','image') for i in irect]

        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]


        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi

    def __getitem__(self, index):

        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index):

        label = deepcopy(self.labels[index])
        label.pop("shape", None)
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self):

        return len(self.labels)

    def update_labels_info(self, label):

        return label

    def build_transforms(self, hyp=None):













        raise NotImplementedError

    def get_labels(self):


















        raise NotImplementedError
    def get_irlabels(self):


















            raise NotImplementedError
