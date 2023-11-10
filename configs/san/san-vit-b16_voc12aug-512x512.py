_base_ = [
    '../_base_/models/san_vit-b16.py',
    '../_base_/datasets/pascal_voc12_aug.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]


train_dataloader = dict(batch_size=6)
# val_dataloader = dict(
#     batch_size=1, dataset=dict(metainfo=metainfo, pipeline=test_pipeline))
# test_dataloader = val_dataloader

# data_preprocessor = dict(
#     mean=[122.7709, 116.7460, 104.0937],
#     std=[68.5005, 66.6322, 70.3232],
#     size_divisor=640,
#     test_cfg=dict(size_divisor=32))
pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/san/clip_vit-base-patch16-224_3rdparty-d08f8887.pth'  #
model = dict(
    # data_preprocessor=data_preprocessor,
    # pretrained='pretrain/vit_base_patch16_224.pth',
    pretrained=pretrained,
    text_encoder=dict(dataset_name='voc'),
    decode_head=dict(num_classes=20))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]
