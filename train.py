import os

import torch
import torch.nn as nn
import torchvision

from matplotlib import pyplot as plt

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


from model import Generator, Discriminator

from tqdm import *
from process_data import get_dataset

if __name__ == '__main__':
    batch_size = 64
    z_dim = 100
    #生成的噪声变量
    z_sample = Variable(torch.randn(100, z_dim)).cuda()
    lr = 1e-4

    """ Medium: WGAN, 50 epoch, n_critic=5, clip_value=0.01 """
    n_epoch = 50# 50
    #判别器训练次数
    n_critic = 5 # 5
    # clip_value = 0.01

    log_dir = os.path.join(r'C:\Users\momo\PycharmProjects\HW6', 'logs')
    ckpt_dir = os.path.join(r'C:\Users\momo\PycharmProjects\HW6', 'checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Model
    G = Generator(in_dim=z_dim).cuda()
    D = Discriminator(3).cuda()
    #设置成为训练模式
    G.train()
    D.train()

    # Loss
    criterion = nn.BCELoss()

    """ Medium: Use RMSprop for WGAN. """
    # Optimizer
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    # opt_D = torch.optim.RMSprop(D.parameters(), lr=lr)
    # opt_G = torch.optim.RMSprop(G.parameters(), lr=lr)


    # DataLoader
    dataset = get_dataset(os.path.join(r'C:\Users\momo\PycharmProjects\HW6', 'faces'))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    steps = 0
    for e, epoch in enumerate(range(n_epoch)):
        progress_bar = tqdm(dataloader)
        for data in progress_bar: #enumerate(progress_bar):    #enumerate(progress_bar):
            imgs = data
            imgs = imgs.cuda()

            bs = imgs.size(0)

            # ============================================
            #  Train D
            # ============================================
            #启用随机噪声发送到gpu上面
            z = Variable(torch.randn(bs, z_dim)).cuda()
            #real'image 这是读取数据集得到的
            r_imgs = Variable(imgs).cuda()
            #fake'image这是将随机噪声输入generater产生的
            f_imgs = G(z)

            """ Medium: Use WGAN Loss. """
            # Label
            #全1向量，代表是真实的
            r_label = torch.ones((bs)).cuda()
            #全零向量，代表fake
            f_label = torch.zeros((bs)).cuda()

            # Model forwarding
            #判别器对于两张图片的判断
            r_logit = D(r_imgs.detach())
            f_logit = D(f_imgs.detach())

            # Compute the loss for the discriminator.
            r_loss = criterion(r_logit, r_label)
            f_loss = criterion(f_logit, f_label)
            loss_D = (r_loss + f_loss) / 2

            # WGAN Loss
            # loss_D = -torch.mean(D(r_imgs)) + torch.mean(D(f_imgs))

            # Model backwarding
            D.zero_grad()
            loss_D.backward()

            # Update the discriminator.
            opt_D.step()

            """ Medium: Clip weights of discriminator. """
            # for p in D.parameters():
            #    p.data.clamp_(-clip_value, clip_value)

            # ============================================
            #  Train G
            # ============================================
            if steps % n_critic == 0:
                # Generate some fake images.
                z = Variable(torch.randn(bs, z_dim)).cuda()
                f_imgs = G(z)

                # Model forwarding
                f_logit = D(f_imgs)

                """ Medium: Use WGAN Loss"""
                # Compute the loss for the generator.
                loss_G = criterion(f_logit, r_label)
                # WGAN Loss
                # loss_G = -torch.mean(D(f_imgs))

                # Model backwarding
                G.zero_grad()
                loss_G.backward()

                # Update the generator.
                opt_G.step()

            steps += 1

            # Set the info of the progress bar
            #   Note that the value of the GAN loss is not directly related to
            #   the quality of the generated images.
            progress_bar.set_description(f"Loss_D:{round(loss_D.item(), 4)}  "
                                         f"Loss_G:{round(loss_G.item(), 4)}  "
                                         f"Epoch:{e + 1}  "
                                         f"Step:{steps}  ")
            # progress_bar.set_infos({
            #     'Loss_D': round(loss_D.item(), 4),
            #     'Loss_G': round(loss_G.item(), 4),
            #     'Epoch': e + 1,
            #     'Step': steps,
            # })

        G.eval()
        f_imgs_sample = (G(z_sample).data + 1) / 2.0
        filename = os.path.join(log_dir, f'Epoch_{epoch + 1:03d}.jpg')
        torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
        print(f' | Save some samples to {filename}.')

        # Show generated images in the jupyter notebook.
        if e % 5 == 0 :
            grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
            plt.figure(figsize=(10, 10))
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()
        G.train()

        if (e + 1) % 5 == 0 or e == 0:
            # Save the checkpoints.
            torch.save(G.state_dict(), os.path.join(ckpt_dir, 'G.pth'))
            torch.save(D.state_dict(), os.path.join(ckpt_dir, 'D.pth'))

