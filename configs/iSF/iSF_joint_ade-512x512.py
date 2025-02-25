_base_ = ["iSF_vit-b16.py", "ade20k.py", "../_base_/default_runtime.py"]


load_from = "work_dirs/iSF_joint_ade-512x512/best_mIoU_iter_22000.pth"
num_classes = 150


# lr=4.5e-4
lr = 1e-4

# train_dataloader = dict(batch_size=28)
train_dataloader = dict(batch_size=10)

iterartions = 60000

tokens = num_classes + 1

ratio = [
    0.1576,
    0.1072,
    0.0878,
    0.0621,
    0.048,
    0.045,
    0.0398,
    0.0231,
    0.0198,
    0.0183,
    0.0181,
    0.0166,
    0.016,
    0.0151,
    0.0118,
    0.011,
    0.0109,
    0.0104,
    0.0104,
    0.0103,
    0.0098,
    0.0074,
    0.0067,
    0.0065,
    0.0061,
    0.006,
    0.0053,
    0.0052,
    0.0046,
    0.0044,
    0.0044,
    0.0044,
    0.0033,
    0.0031,
    0.003,
    0.0027,
    0.0026,
    0.0024,
    0.0024,
    0.0023,
    0.0023,
    0.0022,
    0.0022,
    0.002,
    0.0019,
    0.0019,
    0.0018,
    0.0018,
    0.0018,
    0.0018,
    0.0018,
    0.0018,
    0.0018,
    0.0017,
    0.0017,
    0.0017,
    0.0017,
    0.0017,
    0.0015,
    0.0015,
    0.0015,
    0.0015,
    0.0014,
    0.0014,
    0.0014,
    0.0014,
    0.0014,
    0.0013,
    0.0013,
    0.0013,
    0.0012,
    0.0012,
    0.0012,
    0.0012,
    0.0011,
    0.001,
    0.0009,
    0.0009,
    0.0009,
    0.0009,
    0.0009,
    0.0008,
    0.0008,
    0.0008,
    0.0008,
    0.0008,
    0.0007,
    0.0007,
    0.0007,
    0.0007,
    0.0007,
    0.0007,
    0.0007,
    0.0006,
    0.0006,
    0.0006,
    0.0006,
    0.0006,
    0.0006,
    0.0006,
    0.0006,
    0.0006,
    0.0006,
    0.0006,
    0.0005,
    0.0005,
    0.0005,
    0.0005,
    0.0005,
    0.0005,
    0.0005,
    0.0005,
    0.0005,
    0.0005,
    0.0005,
    0.0005,
    0.0005,
    0.0005,
    0.0005,
    0.0005,
    0.0005,
    0.0004,
    0.0004,
    0.0004,
    0.0004,
    0.0004,
    0.0004,
    0.0004,
    0.0004,
    0.0004,
    0.0004,
    0.0004,
    0.0004,
    0.0004,
    0.0003,
    0.0003,
    0.0003,
    0.0003,
    0.0003,
    0.0003,
    0.0003,
    0.0003,
    0.0003,
    0.0003,
    0.0003,
    0.0003,
    0.0003,
    0.0002,
    0.0002,
    0.0002,
]
# class_weight = [(1 - x) ** 2 for x in ratio]
class_weight = [
    0.001,
    0.7096377600000001,
    0.7970918400000001,
    0.8321088400000001,
    0.8796564099999999,
    0.9063039999999999,
    0.912025,
    0.9219840399999999,
    0.9543336099999999,
    0.96079204,
    0.96373489,
    0.96412761,
    0.9670755600000001,
    0.968256,
    0.9700280099999999,
    0.9765392399999999,
    0.978121,
    0.97831881,
    0.9793081600000001,
    0.9793081600000001,
    0.97950609,
    0.9804960399999999,
    0.9852547600000001,
    0.9866448899999999,
    0.9870422500000001,
    0.98783721,
    0.988036,
    0.9894280900000001,
    0.98962704,
    0.9908211599999999,
    0.9912193600000001,
    0.9912193600000001,
    0.9912193600000001,
    0.99341089,
    0.99380961,
    0.994009,
    0.9946072899999999,
    0.9948067599999999,
    0.99520576,
    0.99520576,
    0.9954052900000001,
    0.9954052900000001,
    0.99560484,
    0.99560484,
    0.996004,
    0.99620361,
    0.99620361,
    0.9964032399999999,
    0.9964032399999999,
    0.9964032399999999,
    0.9964032399999999,
    0.9964032399999999,
    0.9964032399999999,
    0.9964032399999999,
    0.9966028899999999,
    0.9966028899999999,
    0.9966028899999999,
    0.9966028899999999,
    0.9966028899999999,
    0.9970022500000001,
    0.9970022500000001,
    0.9970022500000001,
    0.9970022500000001,
    0.99720196,
    0.99720196,
    0.99720196,
    0.99720196,
    0.99720196,
    0.9974016900000001,
    0.9974016900000001,
    0.9974016900000001,
    0.9976014400000001,
    0.9976014400000001,
    0.9976014400000001,
    0.9976014400000001,
    0.99780121,
    0.998001,
    0.9982008099999999,
    0.9982008099999999,
    0.9982008099999999,
    0.9982008099999999,
    0.9982008099999999,
    0.99840064,
    0.99840064,
    0.99840064,
    0.99840064,
    0.99840064,
    0.99860049,
    0.99860049,
    0.99860049,
    0.99860049,
    0.99860049,
    0.99860049,
    0.99860049,
    0.9988003599999999,
    0.9988003599999999,
    0.9988003599999999,
    0.9988003599999999,
    0.9988003599999999,
    0.9988003599999999,
    0.9988003599999999,
    0.9988003599999999,
    0.9988003599999999,
    0.9988003599999999,
    0.9988003599999999,
    0.9990002500000001,
    0.9990002500000001,
    0.9990002500000001,
    0.9990002500000001,
    0.9990002500000001,
    0.9990002500000001,
    0.9990002500000001,
    0.9990002500000001,
    0.9990002500000001,
    0.9990002500000001,
    0.9990002500000001,
    0.9990002500000001,
    0.9990002500000001,
    0.9990002500000001,
    0.9990002500000001,
    0.9990002500000001,
    0.9990002500000001,
    0.9992001600000001,
    0.9992001600000001,
    0.9992001600000001,
    0.9992001600000001,
    0.9992001600000001,
    0.9992001600000001,
    0.9992001600000001,
    0.9992001600000001,
    0.9992001600000001,
    0.9992001600000001,
    0.9992001600000001,
    0.9992001600000001,
    0.9992001600000001,
    0.9994000900000001,
    0.9994000900000001,
    0.9994000900000001,
    0.9994000900000001,
    0.9994000900000001,
    0.9994000900000001,
    0.9994000900000001,
    0.9994000900000001,
    0.9994000900000001,
    0.9994000900000001,
    0.9994000900000001,
    0.9994000900000001,
    0.9994000900000001,
    0.99960004,
    0.99960004,
    0.99960004,
]

pretrained = "https://download.openmmlab.com/mmsegmentation/v0.5/san/clip_vit-base-patch16-224_3rdparty-d08f8887.pth"  #
model = dict(
    pretrained=pretrained,
    text_encoder=dict(dataset_name="ade"),# cat_bg=False),
    decode_head=dict(
        num_classes=num_classes,
        san_cfg=dict(
            num_queries=tokens,
        ),
        maskgen_cfg=dict(
            sos_token_num=tokens,
        ),
        loss_decode=[
            dict(
                type="CrossEntropyLoss",
                loss_name="loss_cls_ce",
                loss_weight=2.0,
                # class_weight=[1e-2]+[5e-2]+[1e-1]+[1e-1]+[1.0] * (num_classes-4),
                # class_weight=[1e-3] + [1e-1] * 6 + [1.0] * (num_classes - 7),
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
