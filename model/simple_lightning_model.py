import argparse
import os
import torch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint


from utils import (
    build_framewise_valid_mask,
    compute_topk_accuracy,
    configure_validation_evaluator,
    flatten_framewise_logits_and_labels,
)
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
        self.framewise_criterion = torch.nn.CrossEntropyLoss(reduction="none")
        self.validation_evaluator = configure_validation_evaluator(
            head_names=self._get_configured_multi_head_names(),
        )

        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#save-hyperparameters
        self.save_hyperparameters()

    def _maybe_reset_online_memory(self, infos):
        if (
            hasattr(self.model, "clear_memory")
            and len(infos) > 0
            and all(
                info.get("sequence_start", info.get("is_first"))
                for info in infos
            )
        ):
            self.model.clear_memory()

    def _get_memory_keys(self, infos):
        return [
            str(info.get("sequence_id", info.get("sample_id", info["video_id"])))
            for info in infos
        ]

    def on_train_epoch_start(self):
        if hasattr(self.model, "clear_memory"):
            self.model.clear_memory()

    def on_validation_epoch_start(self):
        self.validation_evaluator.reset()
        if hasattr(self.model, "clear_memory"):
            self.model.clear_memory()

    def _split_multi_head_labels(
        self,
        labels: torch.Tensor,
        num_heads: int,
    ) -> list[torch.Tensor]:
        if labels.ndim != 2 or labels.shape[1] != num_heads:
            raise ValueError(
                "Expected labels shaped [batch, num_heads] for multi-head logits, "
                f"but got {tuple(labels.shape)} for {num_heads} heads."
            )
        return list(labels.unbind(dim=1))

    def _get_configured_multi_head_names(self) -> tuple[str, ...] | None:
        if getattr(self.args, "epic_label_type", None) == "verb_noun":
            return ("verb", "noun")
        return None

    def _get_multi_head_metric_names(self, num_heads: int) -> list[str]:
        configured_names = self._get_configured_multi_head_names()
        if configured_names is not None and len(configured_names) == num_heads:
            return list(configured_names)
        return [f"head{head_idx}" for head_idx in range(1, num_heads + 1)]

    def _compute_framewise_loss_and_metrics(self, logits, labels, infos):
        valid_mask = build_framewise_valid_mask(
            infos=infos,
            temporal_dim=labels.shape[1],
            device=logits.device,
        )
        flat_logits, flat_labels = flatten_framewise_logits_and_labels(
            logits=logits,
            labels=labels,
            valid_mask=valid_mask,
        )
        if flat_labels.numel() == 0:
            zero_loss = logits.sum() * 0.0
            return zero_loss, 0.0, 0.0, {}

        per_frame_loss = self.framewise_criterion(flat_logits, flat_labels)
        loss = per_frame_loss.mean()
        top1, top5, *_ = compute_topk_accuracy(flat_logits, flat_labels, topk=(1, 5))
        return loss, top1, top5, {}

    def _compute_loss_and_metrics(self, logits, labels, infos):
        if isinstance(logits, (list, tuple)):
            per_head_labels = self._split_multi_head_labels(labels, len(logits))
            head_losses = []
            top1_values = []
            top5_values = []
            extra_metrics = {}
            head_metric_names = self._get_multi_head_metric_names(len(logits))

            for head_metric_name, (head_logits, head_labels) in zip(
                head_metric_names,
                zip(logits, per_head_labels),
            ):
                head_loss = self.criterion(head_logits, head_labels)
                head_top1, head_top5, *_ = compute_topk_accuracy(
                    head_logits,
                    head_labels,
                    topk=(1, 5),
                )
                head_losses.append(head_loss)
                top1_values.append(head_top1)
                top5_values.append(head_top5)
                extra_metrics[f"loss_{head_metric_name}"] = head_loss.detach()
                extra_metrics[f"top1_{head_metric_name}"] = head_top1
                extra_metrics[f"top5_{head_metric_name}"] = head_top5

            loss = torch.stack(head_losses).mean()
            top1 = sum(top1_values) / len(top1_values)
            top5 = sum(top5_values) / len(top5_values)
            return loss, top1, top5, extra_metrics

        if logits.ndim == 3 and labels.ndim == 2:
            return self._compute_framewise_loss_and_metrics(logits, labels, infos)

        loss = self.criterion(logits, labels)
        top1, top5, *_ = compute_topk_accuracy(logits, labels, topk=(1, 5))
        return loss, top1, top5, {}

    def configure_optimizers(self):
        """see
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """

        optimizer = configure_optimizer(
            optimizer_name=self.args.optimizer_name,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            momentum=self.args.momentum,
            orthogonal_beta=self.args.orthogonal_beta,
            orthogonal_eps=self.args.orthogonal_eps,
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

    def log_train_loss_top15(self, loss, top1, top5, batch_size, extra_metrics=None):
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-epoch-level-metrics
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.log_dict
        metrics = {
            "train_loss": loss.detach(),
            "train_top1": top1,
            "train_top5": top5,
        }
        if extra_metrics:
            metrics.update({
                f"train_{metric_name}": metric_value
                for metric_name, metric_value in extra_metrics.items()
            })
        self.log_dict(
            metrics,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log_dict(
            {
                "train_loss": loss.detach(),
                "train_top1": top1,
                "train_top5": top5,
            },
            prog_bar=True,  # keep the terminal progress bar compact
            logger=False,
            on_step=True,
            on_epoch=False,
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

        data, labels, _, infos = batch
        batch_size = data.size(0)
        video_names = self._get_memory_keys(infos)
        self._maybe_reset_online_memory(infos)

        logits = self.model(data, video_names=video_names)
        loss, top1, top5, extra_metrics = self._compute_loss_and_metrics(
            logits,
            labels,
            infos,
        )
        self.log_train_loss_top15(loss, top1, top5, batch_size, extra_metrics=extra_metrics)

        return loss

    def log_val_loss_top15(self, loss, top1, top5, batch_size, extra_metrics=None):
        metrics = {
            "val_loss": loss.detach(),
            "val_top1": top1,
            "val_top5": top5,
        }
        if extra_metrics:
            metrics.update({
                f"val_{metric_name}": metric_value
                for metric_name, metric_value in extra_metrics.items()
            })
        self.log_dict(
            metrics,
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

        data, labels, _, infos = batch
        batch_size = data.size(0)
        video_names = self._get_memory_keys(infos)
        self._maybe_reset_online_memory(infos)

        logits = self.model(data, video_names=video_names)
        loss, top1, top5, extra_metrics = self._compute_loss_and_metrics(
            logits,
            labels,
            infos,
        )

        if self.validation_evaluator.should_accumulate(infos):
            self.log_val_loss_top15(
                loss,
                top1,
                top5,
                batch_size,
                extra_metrics=extra_metrics,
            )
            self.validation_evaluator.update(logits=logits, labels=labels, infos=infos)
            return

        self.log_val_loss_top15(
            loss,
            top1,
            top5,
            batch_size,
            extra_metrics=extra_metrics,
        )

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
