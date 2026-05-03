import argparse
import os
import torch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint


from utils import compute_topk_accuracy, configure_validation_evaluator
from model import configure_model, ModelConfig
from setup import configure_optimizer, configure_scheduler


class SimpleLightningModel(pl.LightningModule):
    """A simple lightning module

        see
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#methods
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#properties
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks

        https://lightning.ai/docs/pytorch/stable/starter/style_guide.html#method-order
    """

    def __init__(
            self,
            command_line_args: argparse.Namespace,
            exp_name: str,
            cfg=None,
    ):
        """constructor

        see
        https://lightning.ai/docs/pytorch/stable/starter/style_guide.html#init

        Args:
            command_line_args (argparse): args
            n_classes (int): number of categories
            exp_name (str): experiment name of comet.ml
            cfg (optional): config object from defaults.py
        """
        super().__init__()
        self.args = command_line_args
        self.exp_name = exp_name
        self.cfg = cfg

        self.model = configure_model(ModelConfig(
            model_name=self.args.model_name,
            cfg=self.cfg,
            torch_home=self.args.torch_home,
        ))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.validation_evaluator = configure_validation_evaluator()

        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#save-hyperparameters
        self.save_hyperparameters()

    def _maybe_reset_online_memory(self, infos):
        if (
            hasattr(self.model, "clear_memory")
            and len(infos) == 1
            and infos[0].get("sequence_start", infos[0].get("is_first"))
        ):
            self.model.clear_memory()

    def on_validation_epoch_start(self):
        self.validation_evaluator.reset()
        if hasattr(self.model, "clear_memory"):
            self.model.clear_memory()

    def configure_optimizers(self):
        """see
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """

        optimizer = configure_optimizer(
            optimizer_name=self.args.optimizer_name,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            momentum=self.args.momentum,
            model_params=self.model.parameters()
        )
        scheduler = configure_scheduler(
            optimizer=optimizer,
            use_scheduler=self.args.use_scheduler
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def configure_callbacks(self):
        """see
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-callbacks
        """

        save_checkpoint_dir = os.path.join(self.args.save_checkpoint_dir, self.exp_name)
        if self.global_rank == 0:
            os.makedirs(save_checkpoint_dir, exist_ok=True)

        checkpoint_callbacks = [
            ModelCheckpoint(
                dirpath=save_checkpoint_dir,
                monitor="val_top1",
                mode="max",  # larger is better
                save_top_k=2,
                filename="epoch{epoch}_step{step}_acc={val_top1:.2f}",
                auto_insert_metric_name=False,
            ),
        ]

        return checkpoint_callbacks

    def log_train_loss_top15(self, loss, top1, top5, batch_size):
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-epoch-level-metrics
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.log_dict
        self.log_dict(
            {
                "train_loss": loss.item(),
                "train_top1": top1,
            },
            prog_bar=True,  # show on the progress bar
            on_step=True,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,
            batch_size=batch_size,
        )

        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.log
        self.log(
            "train_top5",
            top5,
            prog_bar=False,  # do not show on the progress bar
            on_step=True,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,
            batch_size=batch_size,
        )

    def training_step(self, batch, batch_idx):
        """a single training step for a batch

        Args:
            batch (Tuple[tensor]): a batch of data samples and labels
                (actual type depends on the dataloader)
            batch_idx (int): index of the batch in the epoch

        Returns:
            tensor: loss (used for backward by lightning)

        Note:
            DO NOT USE .to() or model.train() here
                (automatically send to multi-GPUs)
            DO NOT USE loss.backward() here
                (automatically performed by lightning)
            see
                https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-loop
                https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.training_step
        """

        data, labels, frame_indices, infos = batch
        batch_size = data.size(0)
        video_names = [str(m["video_id"]) for m in infos]
        self._maybe_reset_online_memory(infos)

        logits = self.model(data, video_names=video_names)
        loss = self.criterion(logits, labels)

        top1, top5, *_ = compute_topk_accuracy(logits, labels, topk=(1, 5))
        self.log_train_loss_top15(loss, top1, top5, batch_size)

        return loss

    def log_val_loss_top15(self, loss, top1, top5, batch_size):
        self.log_dict(
            {
                "val_loss": loss.item(),
                "val_top1": top1,
                "val_top5": top5,
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,  # sync log metrics for validation
            batch_size=batch_size,
        )

    def validation_step(self, batch, batch_idx):
        """a single validation step for a batch

        Args:
            batch (Tuple[tensor]): a batch of data samples and labels
                (actual type depends on the dataloader)
            batch_idx (int): index of the batch in the epoch

        Note:
            DO NOT USE .to() or model.eval() here
                (automatically send to multi-GPUs)
            DO NOT USE with torch.no_grad() here
                (automatically handled by lightning)
            see
                https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation
                https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.validation_step
        """

        data, labels, frame_indices, infos = batch
        batch_size = data.size(0)
        video_names = [str(m["video_id"]) for m in infos]
        self._maybe_reset_online_memory(infos)

        logits = self.model(data, video_names=video_names)
        loss = self.criterion(logits, labels)

        if self.validation_evaluator.should_accumulate(infos):
            self.log(
                "val_loss",
                loss.item(),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                rank_zero_only=False,
                sync_dist=True,
                batch_size=batch_size,
            )
            self.validation_evaluator.update(logits=logits, labels=labels, infos=infos)
            return

        top1, top5, *_ = compute_topk_accuracy(logits, labels, topk=(1, 5))
        self.log_val_loss_top15(loss, top1, top5, batch_size)

    def on_validation_epoch_end(self):
        metrics = self.validation_evaluator.compute()
        for metric_name, metric_value in metrics.items():
            self.log(
                metric_name,
                metric_value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                rank_zero_only=False,
                sync_dist=False,
            )
