from .image_folder import image_folder, ImageFolderInfo
from .video_folder import video_folder, VideoFolderInfo
from .sequential_video_folder import sequential_video_folder
from .sequential.epic_kitchens_sequential_data_folder import (
    epic_kitchens_sequential_data_folder,
    EpicKitchensSequentialDataFolderInfo,
)
from .sequential.ava_sequential_data_folder import (
    ava_sequential_data_folder,
    AvaSequentialDataFolderInfo,
)

from .transforms import (
    transform_image, TransformImageInfo,
    transform_video, TransformVideoInfo,
    build_epic_kitchens_sequential_transform,
    build_ava_sequential_transform,
)
from .dataloader_factory import configure_dataloader, DataloadersInfo
from .dataset_pl import TrainValDataModule

__all__ = [
    'image_folder',
    'ImageFolderInfo',
    'video_folder',
    'VideoFolderInfo',
    'sequential_video_folder',
    'epic_kitchens_sequential_data_folder',
    'EpicKitchensSequentialDataFolderInfo',
    'ava_sequential_data_folder',
    'AvaSequentialDataFolderInfo',
    'transform_image',
    'TransformImageInfo',
    'transform_video',
    'TransformVideoInfo',
    'build_epic_kitchens_sequential_transform',
    'build_ava_sequential_transform',
    'configure_dataloader',
    'DataloadersInfo',
    'TrainValDataModule',
]


def __getattr__(name):
    if name in {"configure_dataloader", "DataloadersInfo"}:
        from .dataloader_factory import configure_dataloader, DataloadersInfo

        namespace = {
            "configure_dataloader": configure_dataloader,
            "DataloadersInfo": DataloadersInfo,
        }
        return namespace[name]

    if name == "TrainValDataModule":
        from .dataset_pl import TrainValDataModule

        return TrainValDataModule

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
