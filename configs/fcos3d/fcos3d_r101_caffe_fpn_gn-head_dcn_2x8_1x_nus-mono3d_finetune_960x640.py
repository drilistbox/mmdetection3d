_base_ = './fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_huige_setting.py'
# model settings
model = dict(
    train_cfg=dict(
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.05, 0.05]))
# optimizer
optimizer = dict(lr=0.001)
load_from = '/workspace/changyongshu/projects/checkpoints_pretrained/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth'
