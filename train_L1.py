from dataset import VintageFacesDataset
from torch.utils.data import DataLoader
import torch
from unet import UNet
import torch.optim as optim
import torch.nn as nn
import os

train_dir = '/content/sample_data/UTKFaces/train'
val_dir = '/content/sample_data/UTKFaces/val'
img_size = 256
in_chs, out_chs = 1, 2
batch_size = 8
lr = 0.0005
num_epochs = 50
steps_per_epoch = 800
model_weights_path = './model_weights'
train_continue_flag, continue_epoch = True, 27

train_data = VintageFacesDataset(
    img_dir=train_dir, 
    size=img_size
)

val_data = VintageFacesDataset(
    img_dir=val_dir, 
    size=img_size
)

train_dataloader = DataLoader(
    train_data, 
    batch_size=batch_size, 
    shuffle=True
)

val_dataloader = DataLoader(
    val_data, 
    batch_size=batch_size, 
    shuffle=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = UNet(in_channels=in_chs, out_channels=out_chs, init_features=32).to(device)

if train_continue_flag:
    net.load_state_dict(torch.load(os.path.join(model_weights_path, 'epoch_' + str(continue_epoch-1) + '.pth')))

criterion = nn.L1Loss()
optimizer = optim.Adam(
    net.parameters(), 
    lr, 
    betas=(0.5, 0.999)
)

train_loss = []
val_loss = []
best_loss = float('inf')
for epoch in range(continue_epoch, num_epochs):
    net.train(True)
    epoch_loss = 0.0
    batch_idx = 1
    for (x, y) in train_dataloader:

        x = x.to(device)
        y = y.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = net(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        if batch_idx % 50 == 0:
            print('Train Loss:', loss.item())

        if batch_idx == steps_per_epoch: break
        batch_idx += 1

    torch.save(net.state_dict(), os.path.join(model_weights_path, 'epoch_' + str(epoch) + '.pth'))
    train_loss.append(epoch_loss/batch_idx)

    net.train(False)
    with torch.no_grad():
        x, y = next(iter(val_dataloader))
        
        x = x.to(device)
        y = y.to(device)

        # forward pass
        outputs = net(x)
        loss = criterion(outputs, y)
        val_loss.append(loss.item())
        if best_loss > loss.item():
            best_loss = loss.item()
            torch.save(net.state_dict(), os.path.join(model_weights_path, 'early_stopping.pth'))
        
        print('Epoch No:', epoch, ' ', 'Validation Loss:', loss.item())
        print()

        


