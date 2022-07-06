from wgan_gp import WGAN_GP
from dataset import VintageFacesDataset

train_data = VintageFacesDataset(
    img_dir                     = './UTKFaces/train', 
    size                        = 256
)
wgan = WGAN_GP(
    in_channels_gen             = 1,
    init_features_gen           = 32,
    out_channels_gen            = 2,
    in_channels_disc            = 2,
    init_features_disc          = 8
)
wgan.train(
    train_data                  = train_data,
    epochs                      = 25,
    batch_size                  = 8,
    lr_gen                      = 1e-4,
    betas_gen                   = (.9, .99),
    lr_disc                     = 1e-4,
    betas_disc                  = (.9, .99),
    disc_iter                   = 5,
    verbose_every               = 50,
    save_every_epoch            = True,
    save_weights_dir            = './model_weights',
    gp_lambda                   = 10
)