import numpy as np

from data.isaid import RemoveColorMap
from simplecv.api.preprocess import segm, comm


config = dict(
    model=dict(
        type='ResUnet',
        params=dict(
            encoder_config=dict(
                resnet_encoder=dict(
                    resnet_type='resnet50',
                    include_conv5=True,
                    batchnorm_trainable=True,
                    pretrained=True,
                    freeze_at=0,
                    # 8, 16 or 32
                    output_stride=32,
                    with_cp=(False, False, False, False),
                    stem3_3x3=False,
                ),
                aspp_enable=False,
                aspp=dict(
                    in_channel=2048,
                    aspp_dim=256,
                    atrous_rates=(6, 12, 18),
                    add_image_level=True,
                    use_bias=False,
                    use_batchnorm=True,
                    norm_type='batchnorm',
                ),
            ),
            decoder_config=dict(
                type='default',
                params=dict(
                    use_cp=False,
                    in_channels_list=(2048, 1024, 512, 256),
                ),
                cls_in_dim=256,
            ),
            num_classes=16,
            loss=dict(
                ignore_index=255,
                cls_weight=1.0
            )
        )
    ),
    data=dict(
        train=dict(
            type='ISAIDSegmmDataLoader',
            params=dict(
                image_dir='./isaid_segm/train/images',
                mask_dir='./isaid_segm/train/masks',
                patch_config=dict(
                    patch_size=896,
                    stride=512,
                ),
                transforms=[
                    RemoveColorMap(),
                    segm.RandomHorizontalFlip(0.5),
                    segm.RandomVerticalFlip(0.5),
                    segm.RandomRotate90K((0, 1, 2, 3)),
                    segm.FixedPad((896, 896), 255),
                    segm.ToTensor(True),
                    comm.THMeanStdNormalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375))
                ],
                batch_size=4,
                num_workers=2,
                training=True
            ),
        ),
        test=dict(
            type='ISAIDSegmmDataLoader',
            params=dict(
                image_dir='./isaid_segm/val/images',
                mask_dir='./isaid_segm/val/masks',
                patch_config=dict(
                    patch_size=896,
                    stride=512,
                ),
                transforms=[
                    RemoveColorMap(),
                    segm.DivisiblePad(32, 255),
                    segm.ToTensor(True),
                    comm.THMeanStdNormalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375))
                ],
                batch_size=1,
                num_workers=0,
                training=False
            ),
        ),
    ),
    optimizer=dict(
        type='sgd',
        params=dict(
            momentum=0.9,
            weight_decay=0.0001
        ),
        grad_clip=dict(
            max_norm=35,
            norm_type=2,
        )
    ),
    learning_rate=dict(
        type='poly',
        params=dict(
            base_lr=0.007,
            power=0.9,
            max_iters=60000,
        )),
    train=dict(
        forward_times=1,
        num_iters=60000,
        eval_per_epoch=False,
        summary_grads=False,
        summary_weights=False,
        distributed=True,
        apex_sync_bn=True,
        sync_bn=False,
        eval_after_train=True,
        log_interval_step=50,
    ),
    test=dict(
    ),
)