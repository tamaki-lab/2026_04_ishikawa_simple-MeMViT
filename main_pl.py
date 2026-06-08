
import os

import torch
import lightning.pytorch as pl
from lightning.pytorch.plugins import TorchSyncBatchNorm


from args import ArgParse
from logger import configure_logger_pl
from callback import configure_callbacks
from dataset import TrainValDataModule
from model import SimpleLightningModel
from model.memvit.config.defaults import get_cfg, assert_and_infer_cfg


def _parse_lightning_devices(devices_arg: str):
    devices_arg = str(devices_arg).strip()
    if devices_arg == "-1":
        return -1

    requested_devices = [
        int(device.strip()) for device in devices_arg.split(",") if device.strip()
    ]

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        visible_device_ids = [
            device.strip() for device in visible_devices.split(",") if device.strip()
        ]
        requested_device_ids = [str(device) for device in requested_devices]

        # This project historically uses physical GPU IDs on the CLI, while
        # Lightning expects indices relative to CUDA_VISIBLE_DEVICES.
        if requested_device_ids and all(
            device_id in visible_device_ids for device_id in requested_device_ids
        ):
            visible_index_by_id = {
                device_id: index for index, device_id in enumerate(visible_device_ids)
            }
            return [
                visible_index_by_id[device_id] for device_id in requested_device_ids
            ]

    return requested_devices


def _should_use_ddp_find_unused_parameters(devices):
    if devices == -1:
        return torch.cuda.device_count() > 1
    if isinstance(devices, (list, tuple)):
        return len(devices) > 1
    if isinstance(devices, int):
        return devices > 1
    return False


def _configure_model_class_counts(cfg, n_classes):
    if isinstance(n_classes, tuple):
        cfg.MODEL.NUM_CLASSES_LIST = list(n_classes)
        if len(n_classes) > 0:
            cfg.MODEL.NUM_CLASSES = int(n_classes[0])
        return

    cfg.MODEL.NUM_CLASSES_LIST = []
    cfg.MODEL.NUM_CLASSES = int(n_classes)


def main():
    assert torch.cuda.is_available()

    args = ArgParse.get()

    loggers, exp_name = configure_logger_pl(
        model_name=args.model_name,
        disable_logging=args.disable_comet,
        save_dir=args.comet_log_dir,
    )
    data_module = TrainValDataModule(
        command_line_args=args,
        dataset_name=args.dataset_name,
    )

    cfg = get_cfg()
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)

    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    cfg.DETECTION.ENABLE = False
    cfg.DATA.NUM_FRAMES = args.frames_per_clip
    _configure_model_class_counts(cfg, data_module.n_classes)

    cfg = assert_and_infer_cfg(cfg)

    model_lightning = SimpleLightningModel(
        command_line_args=args,
        cfg=cfg,
        exp_name=exp_name
    )
    callbacks = configure_callbacks()
    devices = _parse_lightning_devices(args.devices)
    strategy = (
        "ddp_find_unused_parameters_true"
        if _should_use_ddp_find_unused_parameters(devices)
        else "auto"
    )

    # https://lightning.ai/docs/pytorch/stable/common/trainer.html
    # https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags
    trainer_kwargs = dict(
        devices=devices,
        accelerator="gpu",
        strategy=strategy,
        max_epochs=args.num_epochs,
        logger=loggers,
        log_every_n_steps=args.log_interval_steps,
        accumulate_grad_batches=args.grad_accum,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        plugins=[TorchSyncBatchNorm()],
    )
    if args.val_interval_steps is None:
        trainer_kwargs["check_val_every_n_epoch"] = args.val_interval_epochs
    else:
        trainer_kwargs["val_check_interval"] = args.val_interval_steps

    trainer = pl.Trainer(
        **trainer_kwargs,
        # precision="16-true",  # for FP16 training, use with caution for nan/inf
        # fast_dev_run=True, # only for debug
        # fast_dev_run=5,  # only for debug
        # limit_train_batches=15,  # only for debug
        # limit_val_batches=15,  # only for debug
        # profiler="simple",
    )

    if args.loop_mode == "train":
        trainer.fit(
            model=model_lightning,
            datamodule=data_module,
            ckpt_path=args.checkpoint_to_resume,
        )
    elif args.loop_mode == "val_only":
        trainer.validate(
            model=model_lightning,
            datamodule=data_module,
            ckpt_path=args.checkpoint_to_resume,
        )


if __name__ == "__main__":
    main()
