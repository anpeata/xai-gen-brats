from __future__ import annotations

from typing import Literal

from monai.networks.nets import SegResNet, UNet

ModelName = Literal["unet", "segresnet"]


def create_segmentation_model(model_name: ModelName, in_channels: int = 4, out_channels: int = 4):
    """Factory for 3D segmentation models used in BraTS experiments."""
    if model_name == "unet":
        return UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=0.1,
        )
    if model_name == "segresnet":
        return SegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=16,
            dropout_prob=0.1,
        )
    raise ValueError(f"Unsupported model_name='{model_name}'. Use 'unet' or 'segresnet'.")
