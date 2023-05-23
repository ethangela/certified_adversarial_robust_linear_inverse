
class DefaultConfig(object):

    data_root = './data/DIV2K_gray'
    # label_root = './data/label_128'
    num_data = 3200 #original setting 450000, we use 3200 in this paper
    crop_size = 128
    noise_level = 15

    batch_size = 32  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 1  # how many workers for loading data

    max_epoch = 100
    lr = 0.0005  # initial learning rate
    lr_decay = 0.5

    load_model_path = './checkpoints/DPDNN_denoise_sigma%d.pth'%noise_level
    save_model_path = './checkpoints/DPDNN_denoise_sigma%d.pth'%noise_level

    lrn = 40000
    itr = 6
    eps = 3
    smt = True
    std = 3
    smp = 100
    gpu = '1'
    vis = 0
    pkl = 'test_ord'
    mdl = 'test_ord'


opt = DefaultConfig()


























