from torch.utils.data import DataLoader
import torch
from gan_models import Generator, Discriminator
import torch.optim as optim
import os
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

class WGAN_GP():
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

        gen = Generator(
            in_channels=self.in_channels_gen,
            init_features=self.init_channels_gen,
            out_channels=self.out_channels_gen
        ).to(self.device)

        disc = Discriminator(
            in_channels=self.in_channels_disc,
            init_features=self.init_channels_disc,
        ).to(self.device)

        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}

        return

    def _disc_train_iter(self, X, y):
        
        y_ = self.gen(X)

        # Calculate probabilities on real and generated data
        d_real = self.disc(y)
        d_gen = self.disc(y_)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(y, y_)

        # Create total loss and optimize
        self.optm_disc.zero_grad()
        d_loss = d_gen.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()
        self.optm_disc.step()

        # Record loss
        self.losses['GP'].append(gradient_penalty)
        self.losses['D'].append(d_loss.data[0])

        return

    def _gen_train_iteration(self, X):
        
        self.G_opt.zero_grad()
        y_ = self.gen(X)

        # Calculate loss and optimize
        d_gen = self.D(y_)
        g_loss = -d_gen.mean()
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.losses['G'].append(g_loss)

        return

    def _gradient_penalty(self, y, y_):
        alpha = torch.rand(self.batch_size, 1, 1, 1)
        alpha = alpha.expand_as(y).to(self.device)

        interpolated = alpha * y + (1 - alpha) * y_
        interpolated = Variable(interpolated, requires_grad=True).to(self.device)

        prob_interpolated = self.disc(interpolated)

        gradients = torch_grad(
            outputs=prob_interpolated, 
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()),
            create_graph=True, 
            retain_graph=True
        )[0]

        gradients = gradients.view(self.batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        return self.gp_lambda * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self):
        for i, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)

            self._disc_train_iter(X, y)

            if self.num_steps % self.disc_iters == 0:
                self._gen_train_iter(X)

            if i % self.verbose_every == 0:
                print("Iteration {}".format(i + 1))
                print("D: {}".format(self.losses['D'][-1]))
                print("GP: {}".format(self.losses['GP'][-1]))
                print("Gradient norm: {}".format(self.losses['gradient_norm'][-1]))
                if i > self.disc_iters:
                    print("G: {}".format(self.losses['G'][-1]))

        if self.save_epoch:
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
        
        return

    def train(self,
        train_data              = None,
        epochs                  = 25, 
        batch_size              = 8,
        lr_gen                  = 1e-4,
        betas_gen               = (.9, .99),
        lr_disc                 = 1e-4,
        betas_disc              = (.9, .99),
        disc_iter               = 5,
        verbose_every           = 50,  
        save_every_epoch        = True,
        save_weights_dir        = './model_weights',
        gp_lambda               = 10
    ):

        self.verbose_every      = verbose_every
        self.save_every_epoch   = save_every_epoch
        self.save_weights_dir   = save_weights_dir
        self.disc_iter          = disc_iter
        self.batch_size         = batch_size
        self.gp_lambda          = gp_lambda

        if not os.path.exists(os.path.join(self.save_weights_dir, 'generator')):
            os.makedirs(os.path.join(self.save_weights_dir, 'generator'))
        if not os.path.exists(os.path.join(self.save_weights_dir, 'discriminator')):
            os.makedirs(os.path.join(self.save_weights_dir, 'discriminator'))

        self.optm_gen = optim.Adam(
            self.gen.parameters(),
            lr=lr_gen,
            betas=betas_gen
        )
        
        self.optm_disc = optim.Adam(
            self.disc.parameters(),
            lr=lr_disc,
            betas=betas_disc
        )

        self.train_dataloader = DataLoader(
            train_data, 
            batch_size=self.batch_size, 
            shuffle=True
        )

        for self.epoch in range(1, epochs+1):
            print("\nEpoch {}".format(self.epoch))
            self._train_epoch()
            
        return

    


