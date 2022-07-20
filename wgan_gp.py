from torch.utils.data import DataLoader
import torch
from torch import nn
from gan_models import Generator, Discriminator
import torch.optim as optim
import os
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import pandas as pd
from imageio import imwrite

class WGAN_GP:
    def __init__(self,
        in_channels_gen             = 1, 
        init_features_gen           = 32, 
        out_channels_gen            = 2,
        in_channels_disc            = 2,
        init_features_disc          = 8
    ):

        self.in_channels_gen        = in_channels_gen
        self.init_channels_gen      = init_features_gen
        self.out_channels_gen       = out_channels_gen

        self.in_channels_disc       = in_channels_disc
        self.init_channels_disc     = init_features_disc
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gen = Generator(
            in_channels=self.in_channels_gen,
            init_features=self.init_channels_gen,
            out_channels=self.out_channels_gen
        ).to(self.device)

        self.disc = Discriminator(
            in_channels=self.in_channels_disc,
            init_features=self.init_channels_disc,
        ).to(self.device)

        self.losses = {'G': 0, 'D': 0, 'GP': 0, 'gradient_norm': 0}
        self.losses_df = pd.DataFrame({'G': [], 'D': [], 'GP': [], 'gradient_norm': []})

        return

    def _disc_train_iter(self, X, y):
        
        y_ = self.gen(X)
        d_real = self.disc(y)
        d_gen = self.disc(y_)

        gradient_penalty = self._gradient_penalty(y, y_)

        self.optm_disc.zero_grad()
        d_loss = d_gen.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()
        self.optm_disc.step()

        self.losses['GP'] = gradient_penalty.item()
        self.losses['D'] = d_loss.item()

        return

    def _gen_train_iter(self, X, y):
        
        self.optm_gen.zero_grad()
        y_ = self.gen(X)

        d_gen = self.disc(y_)
        g_loss = ((1 - self.loss_gamma)*(-d_gen.mean())) + (self.loss_gamma*self.L1_criterion(y, y_))
        g_loss.backward()
        self.optm_gen.step()

        self.losses['G'] = g_loss.item()

        return

    def _gradient_penalty(self, y, y_):

        alpha = torch.rand(y.size(dim=0), 1, 1, 1)
        alpha = alpha.expand_as(y).to(self.device)

        interpolated = alpha * y + (1 - alpha) * y_
        interpolated = Variable(interpolated, requires_grad=True).to(self.device)

        prob_interpolated = self.disc(interpolated).to(self.device)

        gradients = torch_grad(
            outputs      = prob_interpolated, 
            inputs       = interpolated,
            grad_outputs = torch.ones(prob_interpolated.size()).to(self.device),
            create_graph =  True, 
            retain_graph = True
        )[0]

        gradients = gradients.view(self.batch_size, -1)
        self.losses['gradient_norm'] = gradients.norm(2, dim=1).mean().item()
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        return self.gp_lambda * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self):
        for i, (X, y) in enumerate(self.train_dataloader):
            
            if i == self.steps_per_epoch:
                break

            X, y = X.to(self.device), y.to(self.device)

            self._disc_train_iter(X, y)

            if i % self.disc_iter == 0:
                self._gen_train_iter(X, y)
                self.losses_df = self.losses_df.append(self.losses, ignore_index=True)
                
            if i % self.verbose_every == 0:
                print("\nIteration {}".format(i))
                print("D: {}".format(self.losses['D']))
                print("GP: {}".format(self.losses['GP']))
                print("Gradient norm: {}".format(self.losses['gradient_norm']))
                if i > self.disc_iter:
                    print("G: {}".format(self.losses['G']))

        if self.save_every_epoch:
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.gen.state_dict(),
                'optimizer_state_dict': self.optm_gen.state_dict()
            }, os.path.join(self.save_weights_dir, 'generator', "Epoch_{}.pth".format(self.epoch)))

            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.disc.state_dict(),
                'optimizer_state_dict': self.optm_disc.state_dict()
            }, os.path.join(self.save_weights_dir, 'discriminator', "Epoch_{}.pth".format(self.epoch)))

            self.losses_df.to_csv('./loss_moniter.csv', index=False)
        return

    def train(self,
        train_data              = None,
        epochs                  = 150, 
        steps_per_epoch         = 1500,
        batch_size              = 8,
        lr_gen                  = 1e-4,
        betas_gen               = (.9, .99),
        lr_disc                 = 1e-4,
        disc_iter               = 5,
        verbose_every           = 50,  
        save_every_epoch        = True,
        save_weights_dir        = './model_weights',
        gp_lambda               = 10,
        loss_gamma              = 0.9,
        continue_epoch          = 12
    ):

        self.verbose_every      = verbose_every
        self.save_every_epoch   = save_every_epoch
        self.save_weights_dir   = save_weights_dir
        self.disc_iter          = disc_iter
        self.batch_size         = batch_size
        self.gp_lambda          = gp_lambda
        self.steps_per_epoch    = steps_per_epoch
        self.loss_gamma         = loss_gamma

        self.optm_gen = optim.Adam(
                self.gen.parameters(),
                lr=lr_gen,
                betas=betas_gen
            )

        self.optm_disc = optim.RMSprop(
            self.disc.parameters(),
            lr=lr_disc
        )

        if continue_epoch != 0:
            gen_path = os.path.join(self.save_weights_dir, 'generator', "Epoch_{}.pth".format(continue_epoch))
            disc_path = os.path.join(self.save_weights_dir, 'discriminator', "Epoch_{}.pth".format(continue_epoch))

            self.gen.load_state_dict(torch.load(gen_path)['model_state_dict'])
            self.disc.load_state_dict(torch.load(disc_path)['model_state_dict'])
            
            self.optm_gen.load_state_dict(torch.load(gen_path)['optimizer_state_dict'])
            self.optm_disc.load_state_dict(torch.load(disc_path)['optimizer_state_dict'])

            self.losses_df = pd.read_csv('./loss_moniter.csv')
        else:
            if not os.path.exists(os.path.join(self.save_weights_dir, 'generator')):
                os.makedirs(os.path.join(self.save_weights_dir, 'generator'))
            if not os.path.exists(os.path.join(self.save_weights_dir, 'discriminator')):
                os.makedirs(os.path.join(self.save_weights_dir, 'discriminator'))

        self.train_dataloader = DataLoader(
            train_data, 
            batch_size=self.batch_size, 
            shuffle=True
        )

        self.L1_criterion = nn.L1Loss()

        self.gen.train(True)
        self.disc.train(True)

        for self.epoch in range(continue_epoch+1, epochs+1):
            print("\nEpoch {}".format(self.epoch))
            self._train_epoch()

        return

    def load_gen(self, gen_path):
        self.gen.load_state_dict(torch.load(gen_path)['model_state_dict'])
        return

    def test(self, test_data, save_result_dir):
        self.test_dataloader = DataLoader(
            test_data, 
            batch_size=1, 
            shuffle=False
        )

        self.gen.train(False)
        
        for i, (X, _) in enumerate(self.test_dataloader):
            X = X.to(self.device)
            with torch.no_grad():
                out_tensor = self.gen(X)
                out = test_data.back_transform(out_tensor)
                imwrite(os.path.join(save_result_dir, str(i)+'.png'), out)

        return

    def evaulate(self):

        return

    


