import argparse


class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.MetavarTypeHelpFormatter
):
    """show default values of argparse.
    see
    https://stackoverflow.com/questions/18462610/argumentparser-epilog-and-description-formatting-in-conjunction-with-argumentdef
    for details.
    """


class ArgParse:

    @staticmethod
    def get() -> argparse.Namespace:
        """generate argparse object

        Returns:
            args (argparse.Namespace): object of command line arguments
        """
        parser = argparse.ArgumentParser(
            description="simple image/video classification",
            formatter_class=CustomFormatter
        )

        # dataset
        parser.add_argument(
            "-r",
            "--root",
            type=str,
            default="./downloaded_data",
            help="root of dataset.",
        )
        parser.add_argument(
            "-d",
            "--dataset_name",
            type=str,
            default="CIFAR10",
            choices=[
                "CIFAR10",
                "ImageFolder",
                "VideoFolder",
                "ZeroImages",
                "SequentialVideoFolder",
                "EpicKitchenSequentialDataset",
                "Salads50SequentialDataset"],
            help="name of dataset.",
        )
        parser.add_argument(
            "-td",
            "--train_dir",
            type=str,
            default="train",
            help="subdir name from root for training set.",
        )
        parser.add_argument(
            "-vd",
            "--val_dir",
            type=str,
            default="val",
            help="subdir name from root for validation set.",
        )
        parser.add_argument(
            "--video_edge_time",
            type=float,
            default=0.5,
            help="number of seconds to ignore the end of the video",
            #       sequential_video_folder
        )
        parser.add_argument(
            "--ext",
            type=str,
            default="*.MP4,*.mp4",
            help="video file extension glob pattern(s). Use comma-separated values for multiple patterns.",
            #       sequential_video_folder
        )
        parser.add_argument(
            "--sequential_label_mode",
            type=str,
            default="video",
            choices=[
                "video",
                "frame"],
            help="label mode for SequentialVideoFolder. video: one label per video/clip. frame: one label per frame in a clip.",
        )
        parser.add_argument(
            "--train_annotation_path",
            type=str,
            default=None,
            help="EPIC-Kitchens train annotation CSV for SequentialVideoFolder frame mode.",
        )
        parser.add_argument(
            "--val_annotation_path",
            type=str,
            default=None,
            help="EPIC-Kitchens validation annotation CSV for SequentialVideoFolder frame mode.",
        )
        parser.add_argument(
            "--epic_label_type",
            type=str,
            default="verb",
            choices=[
                "verb",
                "noun",
                "action",
                "verb_noun"],
            help="EPIC-Kitchens label type to use in sequential datasets. Use verb_noun for paired multi-head verb/noun targets.",
        )
        parser.add_argument(
            "--epic_task",
            type=str,
            default="action_recognition",
            choices=["action_recognition", "action_anticipation"],
            help="EPIC-Kitchens task mode for EpicKitchenSequentialDataset.",
        )
        parser.add_argument(
            "--epic_anticipation_time",
            type=float,
            default=1.0,
            help="seconds of observation gap before the target action for EPIC anticipation.",
        )
        parser.add_argument(
            "--background_label",
            type=str,
            default="background",
            help="fallback label for frames without annotation in SequentialVideoFolder frame mode.",
        )
        parser.add_argument(
            "--label_granularity",
            type=str,
            default="fine",
            choices=["coarse", "fine"],
            help="label granularity for supported sequential datasets such as 50Salads.",
        )
        parser.add_argument(
            "--split_id",
            type=int,
            default=1,
            help="dataset split id for supported sequential datasets such as 50Salads.",
        )
        parser.add_argument(
            "--annotation_root",
            type=str,
            default=None,
            help="root directory containing frame-level annotation files.",
        )
        parser.add_argument(
            "--train_annotation_root",
            type=str,
            default=None,
            help="optional train-only annotation directory. Falls back to --annotation_root when omitted.",
        )
        parser.add_argument(
            "--val_annotation_root",
            type=str,
            default=None,
            help="optional val-only annotation directory. Falls back to --annotation_root when omitted.",
        )
        parser.add_argument(
            "--split_root",
            type=str,
            default=None,
            help="root directory containing train/val split files.",
        )
        parser.add_argument(
            "--train_split_root",
            type=str,
            default=None,
            help="optional train-only split directory. Falls back to --split_root when omitted.",
        )
        parser.add_argument(
            "--val_split_root",
            type=str,
            default=None,
            help="optional val-only split directory. Falls back to --split_root when omitted.",
        )
        parser.add_argument(
            "--label_map_path",
            type=str,
            default=None,
            help="optional label map file to define class ordering explicitly.",
        )
        parser.add_argument(
            "--debug_train_batch_metrics",
            action="store_true",
            help="print detailed metrics for the first training batch and batches with very high train_top1.",
        )
        parser.add_argument(
            "--max_train_clips_per_video",
            type=int,
            default=None,
            help="optional cap on the number of train clips used per video for supported sequential datasets such as 50Salads.",
        )
        # model
        parser.add_argument(
            "--torch_home",
            type=str,
            default="./pretrained_models",
            help="TORCH_HOME environment variable "
            "where pre-trained model weights are stored.",
        )
        parser.add_argument(
            "-m",
            "--model_name",
            type=str,
            default="resnet18",
            choices=["resnet18", "resnet50", "x3d", "abn_r50", "vit_b", "memvit", "zero_output_dummy"],
            help="name of the model",
        )

        parser.add_argument(
            "--use_pretrained",
            dest="use_pretrained",
            action="store_true",
            help="use pretrained model weights (default)",
        )
        parser.add_argument(
            "--scratch",
            dest="use_pretrained",
            action="store_false",
            help="do not use pretrained model weights, "
            "instead train from scratch (not default)",
        )
        parser.set_defaults(use_pretrained=True)

        # video
        parser.add_argument(
            "--frames_per_clip",
            type=int,
            default=16,
            help="frames per clip."
        )
        parser.add_argument(
            "--clip_duration",
            type=float,
            default=80 / 30,
            help="duration of a clip (in second).",
        )
        parser.add_argument(
            "--clips_per_video",
            type=int,
            default=1,
            help="sampling clips per video for validation",
        )

        # training
        parser.add_argument(
            "-b",
            "--batch_size",
            type=int,
            default=8,
            help="batch size."
        )
        parser.add_argument(
            "-w",
            "--num_workers",
            type=int,
            default=2,
            help="number of workers."
        )
        parser.add_argument(
            "-e",
            "--num_epochs",
            type=int,
            default=25,
            help="number of epochs."
        )
        parser.add_argument(
            "-vi",
            "--val_interval_epochs",
            type=int,
            default=1,
            help="validation interval in epochs.",
        )
        parser.add_argument(
            "-li",
            "--log_interval_steps",
            type=int,
            default=1,
            help="logging interval in steps.",
        )

        # optimizer
        parser.add_argument(
            "--optimizer_name",
            type=str,
            default="SGD",
            choices=["SGD", "Adam", "AdamW", "OrthogonalSGD", "OrthogonalAdamW"],
            help="optimizer name.",
        )
        parser.add_argument(
            "--grad_accum",
            type=int,
            default=1,
            help="steps to accumlate gradients.",
        )
        parser.add_argument(
            "-lr",
            type=float,
            default=1e-4,
            help="learning rate."
        )
        parser.add_argument(
            "--momentum",
            type=float,
            default=0.9,
            help="momentum of SGD."
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=5e-4,
            help="weight decay."
        )
        parser.add_argument(
            "--orthogonal_beta",
            type=float,
            default=0.9,
            help="EMA coefficient for orthogonal gradient history.",
        )
        parser.add_argument(
            "--orthogonal_eps",
            type=float,
            default=1e-12,
            help="minimum squared norm for orthogonal projection.",
        )
        parser.add_argument(
            "--use_scheduler",
            dest="use_scheduler",
            action="store_true",
            help="use scheduler (not default)",
        )
        parser.add_argument(
            "--no_scheduler",
            dest="use_scheduler",
            action="store_false",
            help="do not use scheduler (default)",
        )
        parser.set_defaults(use_scheduler=False)

        # multi-GPU strategy
        parser.add_argument(
            "--use_dp",
            dest="use_dp",
            action="store_true",
            help="GPUs with data parallel (dp); not for lightning",
        )
        parser.set_defaults(use_dp=False)

        parser.add_argument(
            "--devices",
            "--gpu_ids",
            type=str,
            default="-1",
            help="GPU ID used for ddp strategy (only for lightning)."
            "\"--devices=0,1\" for 0 and 1, \"--devices=-1\" for all gpus (default).",
            # https://lightning.ai/docs/pytorch/stable/accelerators/gpu_basic.html#choosing-gpu-devices
        )

        # log dirs
        parser.add_argument(
            "--comet_log_dir",
            type=str,
            default="./comet_logs/",
            help="dir to comet log files.",
        )
        parser.add_argument(
            "--tf_log_dir",
            type=str,
            default="./tf_logs/",
            help="dir to TensorBoard log files.",
        )

        # checkpoint files
        parser.add_argument(
            "--save_checkpoint_dir",
            type=str,
            default="./log",
            help="dir to save checkpoint files.",
        )
        parser.add_argument(
            "--checkpoint_to_resume",
            type=str,
            default=None,
            help="path to the checkpoint file to resume from.",
        )

        # disabling comet for debugging
        parser.add_argument(
            "--disable_comet",
            "--no_comet",
            dest="disable_comet",
            action="store_true",
            help="do not use comet.ml (default: use comet)",
        )

        parser.add_argument(
            "--loop_mode",
            type=str,
            default="train",
            choices=["train", "val_only"],
            help="training mode: 'train' for training loop, 'val_only' for validation only.",
        )
        parser.add_argument(
            "-vis",
            "--val_interval_steps",
            type=int,
            default=None,
            help="validation interval in steps.",
        )

        # 以下parser4つ yaml用に追加

        # config file
        parser.add_argument(
            "--cfg_file",
            type=str,
            default=None,
            help="path to config YAML file",
        )

        # config override options
        parser.add_argument(
            "--opts",
            nargs="+",
            type=str,
            default=None,
            help="CLI override options (e.g., --opts TRAIN.BATCH_SIZE 32)",
        )

        # distributed training
        parser.add_argument(
            "--num_shards",
            type=int,
            default=1,
            help="number of shards for distributed training",
        )
        parser.add_argument(
            "--shard_id",
            type=int,
            default=0,
            help="shard id for distributed training",
        )

        parser.set_defaults(disable_comet=False)

        args = parser.parse_args()

        print(args)

        return args
