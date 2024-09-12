from rich.progress import Progress, BarColumn, TextColumn
from rich.console import Console
import torch
import random
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
class Generator(nn.Module):
    def __init__(self, input_channel, output_channels):
        """
        :param input_channel:
        :param output_channels:
        """
        super(Generator, self).__init__()
        self.Conv1 = nn.ConvTranspose2d(input_channel, 512, 4, 2, 1)
        self.Conv2 = nn.ConvTranspose2d(512, 512, 3, 1, 1)
        self.Conv3 = nn.ConvTranspose2d(512, 512, 3, 1, 1)
        self.Conv4 = nn.ConvTranspose2d(512, 256, 3, 1, 1)
        self.Conv5 = nn.ConvTranspose2d(256, 256, 3, 1, 1)
        self.Conv6 = nn.ConvTranspose2d(256, output_channels, 3, 1, 1)
        self.LeakyRelu = nn.LeakyReLU(negative_slope=0.2)
        self.BatchNorm512 = nn.BatchNorm2d(512)
        self.BatchNorm256 = nn.BatchNorm2d(256)
        self.output = nn.Sigmoid()
    def forward(self, x_input):
        x_input = self.Conv1(x_input)
        x_input = self.BatchNorm512(x_input)
        x_input = self.LeakyRelu(x_input)
        x_input = self.Conv2(x_input)
        x_input = self.BatchNorm512(x_input)
        x_input = self.LeakyRelu(x_input)
        x_input = self.Conv3(x_input)
        x_input = self.BatchNorm512(x_input)
        x_input = self.LeakyRelu(x_input)
        x_input = self.Conv4(x_input)
        x_input = self.BatchNorm256(x_input)
        x_input = self.LeakyRelu(x_input)
        x_input = self.Conv5(x_input)
        x_input = self.BatchNorm256(x_input)
        x_input = self.LeakyRelu(x_input)
        x_input = self.Conv6(x_input)
        x_input = self.output(x_input)
        return x_input


channels: int = 3


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512),
            nn.Flatten(),
            # nn.Linear(8192, 2480),
            # nn.LeakyReLU(0.2),
            # nn.Linear(2480, 1024),
            # nn.LeakyReLU(0.2),
            # nn.Linear(1024, 512),
            # nn.LeakyReLU(0.2),
            # nn.Linear(512, 256),
            # nn.LeakyReLU(0.2),
            # nn.Linear(256, 1),
            # nn.Sigmoid()
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        # print(img.shape)
        validity = self.model(img)
        # print(validity.shape, img.shape)
        return validity


# x = torch.randn(2, 512, 64, 64, requires_grad=False)
# discrimination = Generator(512, 3)
# print(discrimination(x).shape)
# mnop = np.array(discrimination(x).detach())
# plt.imshow(mnop[0].transpose(1, 2, 0))
# plt.show()
def training(generator_: nn.Module, discriminator_: nn.Module, real_inputs, epochs: int, batch_size: int,
             learning_rate: float):
    optimizer_g = torch.optim.AdamW(generator_.parameters(), lr=learning_rate)
    optimizer_d = torch.optim.AdamW(discriminator_.parameters(), lr=learning_rate)
    loss1 = nn.BCELoss()
    for i in range(epochs):
        # description = tqdm.set_description_str(f'Epoch {i + 1}/{epochs}')
        for j in range(real_inputs.shape[0]):
            discriminator_.zero_grad()
            # real_inputs = real_inputs[j * batch_size:(j + 1) * batch_size]
            real_inputs = real_inputs.to('cpu')
            real_labels = torch.ones((batch_size, 1)).to('cpu')
            fake_labels = torch.zeros((batch_size, 1)).to('cpu')
            # print(1)
            fake_inputs = generator_(torch.randn(batch_size, 512, 32, 32).to('cpu'))
            # print(1.5)
            logits_1 = discriminator_(real_inputs)
            real_loss = loss1(logits_1, real_labels)
            real_loss.backward()
            logits_2 = discriminator_(fake_inputs)
            fake_loss = loss1(logits_2, fake_labels)
            fake_loss.backward()
            # print(1.7)
            optimizer_d.step()
            generator_.zero_grad()
            fake_label1 = discriminator_(fake_inputs)
            loss_g = loss1(fake_label1, real_labels)
            loss_g.backward()
            optimizer_g.step()
            print(loss_g.item(), real_loss.item() + fake_loss.item())


generator = Generator(512, 3)
discriminator = Discriminator()
real_input = torch.randn(2, 3, 64, 64).to('cpu')  # Mockup real input images

training(generator, discriminator, real_input, epochs=10, batch_size=2, learning_rate=0.0002)



# def training(generator_: nn.Module, discriminator_: nn.Module, real_inputs, epochs: int, batch_size: int,
#              learning_rate: float):
#     optimizer_g = torch.optim.AdamW(generator_.parameters(), lr=learning_rate)
#     optimizer_d = torch.optim.AdamW(discriminator_.parameters(), lr=learning_rate)
#     loss1 = nn.BCELoss()
#     for i in range(epochs):
#         for j in range(real_inputs.shape[0] // batch_size):
#             discriminator_.zero_grad()
#             real_batch = real_inputs[j * batch_size:(j + 1) * batch_size]
#             real_batch = real_batch.to('cpu')
#             real_labels = torch.ones((batch_size, 1)).to('cpu')
#             fake_labels = torch.zeros((batch_size, 1)).to('cpu')
#             fake_inputs = generator_(torch.randn(batch_size, 512, 32, 32).to('cpu'))
#
#             logits_real = discriminator_(real_batch)
#             real_loss = loss1(logits_real, real_labels)
#             real_loss.backward()
#
#             logits_fake = discriminator_(fake_inputs.detach())
#             fake_loss = loss1(logits_fake, fake_labels)
#             fake_loss.backward()
#             optimizer_d.step()
#
#             generator_.zero_grad()
#             fake_inputs = generator_(torch.randn(batch_size, 512, 32, 32).to('cpu'))
#             fake_label1 = discriminator_(fake_inputs)
#             loss_g = loss1(fake_label1, real_labels)
#             loss_g.backward()  # Use retain_graph=True if necessary
#             optimizer_g.step()
#
#             print(f"Epoch [{i+1}/{epochs}] Batch [{j+1}/{real_inputs.shape[0] // batch_size}] "
#                   f"Loss_G: {loss_g.item()} Loss_D: {real_loss.item() + fake_loss.item()}")