from wgan_gp import WGAN_GP
from dataset import VintageFacesDataset

train_data = VintageFacesDataset(
    img_dir                     = '/content/sample_data/UTKFaces/train', 
    size                        = 256
)

wgan_gp = WGAN_GP(
    in_channels_gen             = 1,
    init_features_gen           = 32,
    out_channels_gen            = 3,
    in_channels_disc            = 3,
    init_features_disc          = 8
)

wgan_gp.train(
    train_data                  = train_data,
    epochs                      = 55,
    steps_per_epoch             = 1500,
    batch_size                  = 8,
    lr_gen                      = 1e-4,
    betas_gen                   = (.5, .99),
    lr_disc                     = 1e-4,
    betas_disc                  = (.5, .99),
    disc_iter                   = 5,
    verbose_every               = 50,
    save_every_epoch            = True,
    save_weights_dir            = './model_weights',
    gp_lambda                   = 10,
    loss_gamma                  = 0.7,
    continue_epoch              = 0
)