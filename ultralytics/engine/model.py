import inspect
from pathlib import Path
from typing import Union

import numpy as np
import torch

from ultralytics.cfg import TASK2DATA, get_cfg, get_save_dir
HUB_WEB_ROOT = "https://hub.ultralytics.com"
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, nn, yaml_model_load
from ultralytics.utils import (
    ARGV,
    ASSETS,
    DEFAULT_CFG_DICT,
    LOGGER,
    RANK,
    SETTINGS,
    callbacks,
    checks,
    emojis,
    yaml_load,
)


class Model(nn.Module):


































































    def __init__(
        self,
        model: Union[str, Path] = "yolov8n.pt",
        task: str = None,
        verbose: bool = False,
    ) -> None:




















        super().__init__()
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None
        self.model = None
        self.trainer = None
        self.ckpt = None
        self.cfg = None
        self.ckpt_path = None
        self.overrides = {}
        self.metrics = None
        self.session = None
        self.task = task
        model = str(model).strip()


        if self.is_hub_model(model):

            checks.check_requirements("hub-sdk>=0.0.6")
            self.session = self._get_hub_session(model)
            model = self.session.model_file


        elif self.is_triton_model(model):
            self.model_name = self.model = model
            self.task = task
            return


        if Path(model).suffix in {".yaml", ".yml"}:
            self._new(model, task=task, verbose=verbose)
        else:
            self._load(model, task=task)

    def __call__(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs,
    ) -> list:

















        return self.predict(source, stream, **kwargs)

    @staticmethod
    def _get_hub_session(model: str):

        from ultralytics.hub.session import HUBTrainingSession

        session = HUBTrainingSession(model)
        return session if session.client.authenticated else None

    @staticmethod
    def is_triton_model(model: str) -> bool:

        from urllib.parse import urlsplit

        url = urlsplit(model)
        return url.netloc and url.path and url.scheme in {"http", "grpc"}

    @staticmethod
    def is_hub_model(model: str) -> bool:

        return any(
            (
                model.startswith(f"{HUB_WEB_ROOT}/models/"),
                [len(x) for x in model.split("_")] == [42, 20],
                len(model) == 20 and not Path(model).exists() and all(x not in model for x in "./\\"),
            )
        )

    def _new(self, cfg: str, task=None, model=None, verbose=False) -> None:









        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)
        self.model = (model or self._smart_load("model"))(cfg_dict, verbose=verbose and RANK == -1)
        self.overrides["model"] = self.cfg
        self.overrides["task"] = self.task


        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}
        self.model.task = self.task
        self.model_name = cfg

    def _load(self, weights: str, task=None) -> None:







        if weights.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):
            weights = checks.check_file(weights)
        weights = checks.check_model_file_from_stem(weights)

        if Path(weights).suffix == ".pt":
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = self.model.args["task"]
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            weights = checks.check_file(weights)
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides["model"] = weights
        self.overrides["task"] = self.task
        self.model_name = weights

    def _check_is_pytorch_model(self) -> None:

        pt_str = isinstance(self.model, (str, Path)) and Path(self.model).suffix == ".pt"
        pt_module = isinstance(self.model, nn.Module)
        if not (pt_module or pt_str):
            raise TypeError(
                f"model='{self.model}' should be a *.pt PyTorch model to run this method, but is a different format. "
                f"PyTorch models can train, val, predict and export, i.e. 'model.train(data=...)', but exported "
                f"formats like ONNX, TensorRT etc. only support 'predict' and 'val' modes, "
                f"i.e. 'yolo predict model=yolov8n.onnx'.\nTo run CUDA or MPS inference please pass the device "
                f"argument directly in your inference command, i.e. 'model.predict(source=..., device=0)'"
            )

    def reset_weights(self) -> "Model":













        self._check_is_pytorch_model()
        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True
        return self

    def load(self, weights: Union[str, Path] = "yolov8n.pt") -> "Model":















        self._check_is_pytorch_model()
        if isinstance(weights, (str, Path)):
            weights, self.ckpt = attempt_load_one_weight(weights)
        self.model.load(weights)
        return self

    def save(self, filename: Union[str, Path] = "saved_model.pt", use_dill=True) -> None:












        self._check_is_pytorch_model()
        from datetime import datetime

        from ultralytics import __version__

        updates = {
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }
        torch.save({**self.ckpt, **updates}, filename, use_dill=use_dill)

    def info(self, detailed: bool = False, verbose: bool = True):
















        self._check_is_pytorch_model()
        return self.model.info(detailed=detailed, verbose=verbose)

    def fuse(self):








        self._check_is_pytorch_model()
        self.model.fuse()

    def embed(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs,
    ) -> list:


















        if not kwargs.get("embed"):
            kwargs["embed"] = [len(self.model.model) - 2]
        return self.predict(source, stream, **kwargs)

    def predict(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        predictor=None,
        obb=False,
        **kwargs,
    ) -> list:




























        if source is None:
            source = ASSETS
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")

        is_cli = (ARGV[0].endswith("yolo") or ARGV[0].endswith("ultralytics")) and any(
            x in ARGV for x in ("predict", "track", "mode=predict", "mode=track")
        )

        custom = {"conf": 0.25, "batch": 1, "save": is_cli, "mode": "predict"}
        args = {**self.overrides, **custom, **kwargs}
        prompts = args.pop("prompts", None)



        if not self.predictor:
            self.predictor = predictor or self._smart_load("predictor")(overrides=args, _callbacks=self.callbacks)


            self.predictor.setup_model(model=self.model, verbose=is_cli)

        else:
            self.predictor.args = get_cfg(self.predictor.args, args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        if prompts and hasattr(self.predictor, "set_prompts"):
            self.predictor.set_prompts(prompts)


        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)

    def track(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        persist: bool = False,
        **kwargs,
    ) -> list:
























        if not hasattr(self.predictor, "trackers"):
            from ultralytics.trackers import register_tracker

            register_tracker(self, persist)
        kwargs["conf"] = kwargs.get("conf") or 0.1
        kwargs["batch"] = kwargs.get("batch") or 1
        kwargs["mode"] = "track"
        return self.predict(source=source, stream=stream, **kwargs)

    def val(
        self,
        validator=None,
        **kwargs,

    ):
























        custom = {"rect": True}
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def benchmark(
        self,
        **kwargs,
    ):






















        self._check_is_pytorch_model()
        from ultralytics.utils.benchmarks import benchmark

        custom = {"verbose": False}
        args = {**DEFAULT_CFG_DICT, **self.model.args, **custom, **kwargs, "mode": "benchmark"}
        return benchmark(
            model=self,
            data=kwargs.get("data"),
            imgsz=args["imgsz"],
            half=args["half"],
            int8=args["int8"],
            device=args["device"],
            verbose=kwargs.get("verbose"),
        )

    def export(
        self,
        **kwargs,
    ):




















        self._check_is_pytorch_model()
        from .exporter import Exporter

        custom = {"imgsz": self.model.args["imgsz"], "batch": 1, "data": None, "verbose": False}
        args = {**self.overrides, **custom, **kwargs, "mode": "export"}
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)

    def train(
        self,
        trainer=None,
        **kwargs,
    ):



























        self._check_is_pytorch_model()
        if hasattr(self.session, "model") and self.session.model.id:
            if any(kwargs):
                LOGGER.warning("WARNING ⚠️ using HUB training arguments, ignoring local training arguments.")
            kwargs = self.session.train_args

        checks.check_pip_update_available()

        overrides = yaml_load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides
        custom = {

            "data": overrides.get("data") or DEFAULT_CFG_DICT["data"] or TASK2DATA[self.task],
            "model": self.overrides["model"],
            "task": self.task,
        }
        args = {**overrides, **custom, **kwargs, "mode": "train"}
        if args.get("resume"):
            args["resume"] = self.ckpt_path

        self.trainer = (trainer or self._smart_load("trainer"))(overrides=args, _callbacks=self.callbacks)
        if not args.get("resume"):
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model

            if SETTINGS["hub"] is True and not self.session:

                try:
                    self.session = self._get_hub_session(self.model_name)
                    if self.session:
                        self.session.create_model(args)

                        if not getattr(self.session.model, "id", None):
                            self.session = None
                except (PermissionError, ModuleNotFoundError):

                    pass

        self.trainer.hub_session = self.session
        self.trainer.train()

        if RANK in {-1, 0}:
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, _ = attempt_load_one_weight(ckpt)
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, "metrics", None)
        return self.metrics

    def tune(
        self,
        use_ray=False,
        iterations=10,
        *args,
        **kwargs,
    ):




















        self._check_is_pytorch_model()
        if use_ray:
            from ultralytics.utils.tuner import run_ray_tune

            return run_ray_tune(self, max_samples=iterations, *args, **kwargs)
        else:
            from .tuner import Tuner

            custom = {}
            args = {**self.overrides, **custom, **kwargs, "mode": "train"}
            return Tuner(args=args, _callbacks=self.callbacks)(model=self, iterations=iterations)

    def _apply(self, fn) -> "Model":

        self._check_is_pytorch_model()
        self = super()._apply(fn)
        self.predictor = None
        self.overrides["device"] = self.device
        return self

    @property
    def names(self) -> list:









        from ultralytics.nn.autobackend import check_class_names

        if hasattr(self.model, "names"):
            return check_class_names(self.model.names)
        else:
            if not self.predictor:
                self.predictor = self._smart_load("predictor")(overrides=self.overrides, _callbacks=self.callbacks)
                self.predictor.setup_model(model=self.model, verbose=False)
            return self.predictor.model.names

    @property
    def device(self) -> torch.device:









        return next(self.model.parameters()).device if isinstance(self.model, nn.Module) else None

    @property
    def transforms(self):








        return self.model.transforms if hasattr(self.model, "transforms") else None

    def add_callback(self, event: str, func) -> None:













        self.callbacks[event].append(func)

    def clear_callback(self, event: str) -> None:











        self.callbacks[event] = []

    def reset_callbacks(self) -> None:






        for event in callbacks.default_callbacks.keys():
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]

    @staticmethod
    def _reset_ckpt_args(args: dict) -> dict:

        include = {"imgsz", "data", "task", "single_cls"}
        return {k: v for k, v in args.items() if k in include}






    def _smart_load(self, key: str):

        try:
            return self.task_map[self.task][key]
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]
            raise NotImplementedError(
                emojis(f"WARNING ⚠️ '{name}' model does not support '{mode}' mode for '{self.task}' task yet.")
            ) from e

    @property
    def task_map(self) -> dict:






        raise NotImplementedError("Please provide task map for your model!")
