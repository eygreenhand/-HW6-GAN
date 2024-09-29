import os

import torchvision

import matplotlib.pyplot as plt

from process_data import get_dataset

dataset = get_dataset(os.path.join(r'C:\Users\momo\PycharmProjects\HW6', 'faces'))

images = [dataset[i] for i in range(16)]
grid_img = torchvision.utils.make_grid(images, nrow=4)
plt.figure(figsize=(10,10))
plt.imshow(grid_img.permute(1, 2, 0))
plt.show()


images = [(dataset[i]+1)/3 for i in range(16)]
grid_img = torchvision.utils.make_grid(images, nrow=4)
plt.figure(figsize=(10,10))
plt.imshow(grid_img.permute(1, 2, 0))
plt.show()


