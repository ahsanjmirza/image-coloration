import torch
import torch.nn as nn
import torch.optim as optim
from model import Generator, Discriminator
from data_utils import train_data_generator, get_steps_per_epoch
import os

# Hyperparameters
lr = 0.0005
batch_size = 8
image_size = 256
channels_img_in = 1
channels_img_out = 2
num_epochs = 50
features_g = 8
discriminator_updates_mod = 2
dataset_path = './dataset'
continue_flag, from_epoch = False, 0
steps_per_epoch = get_steps_per_epoch(dataset_path)
model_weights_save_path = './model_weights'

data_loader = train_data_generator(dataset_path, continue_flag)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create discriminator and generator
if continue_flag:
    netD = torch.load(model_weights_save_path + "/discriminator/" + str(from_epoch-1) + '.pth')
    netG = torch.load(model_weights_save_path + "/generator/" + str(from_epoch-1) + '.pth')
    print("Training Restarted")
else:
    netD = Discriminator(channels_img_out).to(device)
    netG = Generator(channels_img_in, features_g, channels_img_out).to(device)
    print("Training Started")

# Setup Optimizer for G and D
optimizerD = optim.Adam(netD.parameters(), lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr, betas=(0.5, 0.999))

netG.train()
netD.train()

criterion = nn.BCELoss()

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(data_loader):
        
        data = torch.from_numpy(data).permute(0, 3, 1, 2).to(device)
        targets = torch.from_numpy(targets).permute(0, 3, 1, 2).to(device)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        if batch_idx % discriminator_updates_mod == 0:
            netD.zero_grad()
            output = netD(targets).reshape(-1)
            label = (torch.ones(output.shape)*0.9).to(device)
            lossD_real = criterion(output, label)
            lossD_real.backward()

            fake = netG(data)
            output = netD(fake.detach()).reshape(-1)
            label = (torch.ones(output.shape)*0.1).to(device)
            lossD_fake = criterion(output, label)
            lossD_fake.backward()

            lossD = (lossD_real + lossD_fake) 
            optimizerD.step()
        else: fake = netG(data)

        ### Train Generator: max log(D(G(z)))
        netG.zero_grad()
        output = netD(fake).reshape(-1)
        label = torch.ones(output.shape).to(device)
        lossG = criterion(output, label)
        lossG.backward()
        optimizerG.step()

        if batch_idx == steps_per_epoch: break
        if batch_idx % 50 == 0: 
            print("Epoch:", epoch, "Batches_done: ", batch_idx)
            print("D_loss:", lossD/2)
            print("G_loss:", lossG)
            print()

    torch.save(netG, model_weights_save_path + "/generator/" + str(epoch+from_epoch) + '.pth')
    torch.save(netD, model_weights_save_path + "/discriminator/" + str(epoch+from_epoch) + '.pth')
        



