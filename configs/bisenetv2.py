
## bisenetv2
cfg = dict(
    model_type='bisenetv2',
    num_aux_heads=4,
    lr_start = 5e-2,
    weight_decay=5e-4,
    warmup_iters = 1000,
    # max_iter = 150000,
    # im_root='./datasets/cityscapes',
    # train_img_anns='./datasets/cityscapes/train.txt',
    # val_img_anns='./datasets/cityscapes/val.txt',
    train_img_root='./dataset/carla/dataABC/CameraRGB',
    train_img_anns='./dataset/carla/dataABC/CameraSeg',
    val_img_root='./dataset/carla/dataD/CameraRGB',
    val_img_anns='./dataset/carla/dataD/CameraSeg',
    scales=[0.5, 2.], #[0.25, 2.],
    cropsize=[600, 800], #[360, 640], #[512, 1024], ## training size
    imgs_per_gpu=12,  ## batch_size / gpu_count #8,
    use_fp16=False, #True,
    use_sync_bn=False,
    respth='./res',
    input_size=None,  ## aspect=9:16 #[600,800], #[512, 1024],
    ## Don't change this param, input_size. Sould be always None.
    ## If you set this param, input image is resized and that aspect is changed so model predict accuracy get wrong.
    random_seed=123,
    n_classes=13, #23,
    anns_ignore=255, #0,
    max_epoch=500,  # cfg.max_iter / len(dl.dataset) * cfg.imgs_per_gpu
    log_level='info',  ##{'debug', 'info', 'warning', 'error', 'critical'}
    weight_path='./res/weight',
)
