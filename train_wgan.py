from wgan_gp import WGAN_GP
from dataset import VintageFacesDataset

import warnings
warnings.filterwarnings('ignore')

train_data = VintageFacesDataset(
    img_dir                     = './dataset',
    size                        = 256
)

wgan_gp = WGAN_GP(
    in_channels_gen             = 1,
    init_features_gen           = 64,
    out_channels_gen            = 3,
    in_channels_disc            = 3,
    init_features_disc          = 8
)

wgan_gp.train(
    train_data                  = train_data,
    epochs                      = 200,
    steps_per_epoch             = 2000,
    batch_size                  = 8,
    lr_gen                      = 1e-4,
    lr_disc                     = 1e-4,
    disc_iter                   = 5,
    verbose_every               = 50,
    save_every_epoch            = True,
    save_weights_dir            = './model_weights',
    gp_lambda                   = 10,
    loss_gamma                  = 0.7,
    continue_epoch              = 0
)