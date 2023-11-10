_base_ = ["iSF_vit-b16.py", "pascal_voc12_aug.py", "../_base_/default_runtime.py"]


load_from = "/media/caesarg/Game/Master/Research/Incremental Learning/test/mmsegmentation/work_dirs/iSF_joint_voc12aug-512x512/best_mIoU_iter_250.pth"

num_classes = 20


# lr=4.5e-4
lr = 1e-4

# train_dataloader = dict(batch_size=28)
train_dataloader = dict(batch_size=6)

iterartions = 600000


class_weight = [
    0.0670279711484909,
    0.9878897070884705,
    0.9947828054428101,
    0.9847723245620728,
    0.9905310869216919,
    0.9883438944816589,
    0.97190922498703,
    0.9772062301635742,
    0.9580444097518921,
    0.984828770160675,
    0.9838433265686035,
    0.9804360270500183,
    0.9682194590568542,
    0.9833130836486816,
    0.9817487001419067,
    0.9204536080360413,
    0.9905127882957458,
    0.985997200012207,
    0.9794195294380188,
    0.9748967289924622,
    0.9866529703140259,
    0.9139423370361328,
]

pretrained = "https://download.openmmlab.com/mmsegmentation/v0.5/san/clip_vit-base-patch16-224_3rdparty-d08f8887.pth"  #
model = dict(
    pretrained=pretrained,
    text_encoder=dict(dataset_name="voc"),
    decode_head=dict(
        num_classes=num_classes,
        loss_decode=[
            dict(
                type="CrossEntropyLoss",
                loss_name="loss_cls_ce",
                loss_weight=2.0,
                class_weight=class_weight,
            ),
            dict(
                type="CrossEntropyLoss",
                use_sigmoid=True,
                loss_name="loss_mask_ce",
                loss_weight=5.0,
            ),
            dict(
                type="DiceLoss",
                ignore_index=None,
                naive_dice=True,
                eps=1,
                loss_name="loss_mask_dice",
                loss_weight=5.0,
            ),
        ],
    ),
)


# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=lr, betas=(0.9, 0.999), weight_decay=1e-4),
    paramwise_cfg=dict(
        custom_keys={
            "pos_embed": dict(decay_mult=0.0),
            "cls_token": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

param_scheduler = [
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=0.9,
        begin=0,
        end=iterartions,
        by_epoch=False,
    )
]


# training schedule for 60k
train_cfg = dict(type="IterBasedTrainLoop", max_iters=iterartions, val_interval=2000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")    
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=2000,
        save_last=True,
        save_best="mIoU",
        rule="greater",
        max_keep_ckpts=10,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
