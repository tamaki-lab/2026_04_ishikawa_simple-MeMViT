from torchvision import transforms
from torchvision.transforms import (
    CenterCrop,
    RandomCrop,
    RandomHorizontalFlip,
    Compose,
)

from pytorchvideo.transforms import (
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample,
)


def build_sequential_video_transform(
    is_train: bool,
    frames_per_clip: int = 16,
):
    """
    Sequential Video Dataset 用の動画 transform を作成する．
    入力想定:
        clip: torch.Tensor
        shape: C x T x H x W

    処理内容:
        1. 時間方向を frames_per_clip に揃える
        2. pixel値を 0〜1 に正規化する
        3. ImageNet mean/std で正規化する
        4. train / val に応じた空間方向の augmentation を行う

    Parameters
    ----------
    is_train：学習用 transform or 検証用 transform
    frames_per_clip：1 clip に含めるフレーム数
    """

    transform_list = [
        UniformTemporalSubsample(frames_per_clip),
        transforms.Lambda(lambda x: x / 255.0),
        Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]

    if is_train:
        transform_list.extend(
            [
                RandomShortSideScale(
                    min_size=256,
                    max_size=320,
                ),
                RandomCrop(224),
                RandomHorizontalFlip(),
            ]
        )
    else:
        transform_list.extend(
            [
                ShortSideScale(256),
                CenterCrop(224),
            ]
        )

    return Compose(transform_list)


# 既存コードの transform_video という名前を使っていた箇所との互換用．
def transform_video(
    is_train: bool,
    frames_per_clip: int = 16,
):
    return build_sequential_video_transform(
        is_train=is_train,
        frames_per_clip=frames_per_clip,
    )
