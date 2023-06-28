from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import imageio

cudnn.benchmark = True

# set manual seed to a constant get a consistent output
manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# for converting the image to PIL image
to_pil_image = transforms.ToPILImage()

# load the dataset
dataset = dset.MNIST(root='./data', download=True,
                       transform=transforms.Compose([
                           transforms.Resize(28), # resizing the image to 28x28
                           transforms.ToTensor(), # converting the image to tensor
                           transforms.Normalize((0.5,), (0.5,)), # normalizing the image to mean=0.5 and std=0.5
                       ]))

# checking the availability of cuda devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# setting parameters
num_channels = 1 # number of channels in image(since the image is grayscale the number of channels are 1)
ngpu = 1 # number of gpu's available
nz = 100 # input noise dimension
gen_filters = 64 # number of generator filters
disc_filters = 64 # number of discriminator filters
batch_size = 64 # batch size during training
learning_rate = 0.0002 # learning rate for optimizers
num_epochs = 20 # number of epochs for training

# creating the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=2)

# custom weights initialization called on generator and discriminator
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: # if the layer is convolutional layer
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1: # if the layer is batch normalization layer
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# generator class
class Generator(nn.Module):
    def __init__(self, ngpu, num_channels, nz, gen_filters):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # starting block with input noise dimension as input
            nn.ConvTranspose2d(nz, gen_filters * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(gen_filters * 8),
            nn.ReLU(True),
            # second block
            nn.ConvTranspose2d(gen_filters * 8, gen_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_filters * 4),
            nn.ReLU(True),
            # third block
            nn.ConvTranspose2d(gen_filters * 4, gen_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_filters * 2),
            nn.ReLU(True),
            # fourth block
            nn.ConvTranspose2d(gen_filters * 2, gen_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_filters),
            nn.ReLU(True),
            # output block
            nn.ConvTranspose2d(gen_filters, num_channels, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh() # using tanh activation function for the last layer
        )

    # forward pass function
    def forward(self, input):
        if input.is_cuda and self.ngpu > 1: # if the input is on gpu and the number of gpu's available is greater than 1
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# creating the generator
generator = Generator(ngpu,num_channels,nz,gen_filters).to(device)
# initializing the weights of the generator
generator.apply(init_weights)
print(generator)

# discriminator class
class Discriminator(nn.Module):
    def __init__(self, ngpu, num_channels, disc_filters):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input block with input image dimension as input
            nn.Conv2d(num_channels, disc_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # second block
            nn.Conv2d(disc_filters, disc_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(disc_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # third block
            nn.Conv2d(disc_filters * 2, disc_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(disc_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # output block
            nn.Conv2d(disc_filters * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid() # using sigmoid activation function for the last layer
        )

    # forward pass function
    def forward(self, input):
        if input.is_cuda and self.ngpu > 1: # if the input is on gpu and the number of gpu's available is greater than 1
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)

# creating the discriminator
discriminator = Discriminator(ngpu,num_channels,disc_filters).to(device)
# initializing the weights of the discriminator
discriminator.apply(init_weights)
print(discriminator)

# define binary cross entropy as loss function
loss_function = nn.BCELoss()

# define optimizer for generator and discriminator
discriminator_opt = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
generator_opt = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

fixed_noise = torch.randn(64, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

gen_losses = [] # to store generator loss after each epoch
disc_losses = [] # to store discriminator loss after each epoch
images = [] # to store images generated by the generator

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):

        # train discriminator with real images
        discriminator.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device, dtype=torch.float)

        # forward pass with real images
        output = discriminator(real_cpu)
        disc_error_real = loss_function(output, label)
        disc_error_real.backward()
        D_x = output.mean().item()

        # train discriminator with fake images
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = generator(noise)
        label.fill_(fake_label)

        # forward pass with fake images
        output = discriminator(fake.detach())
        disc_error_fake = loss_function(output, label)
        disc_error_fake.backward()
        D_G_z1 = output.mean().item()

        # calculate discriminator loss and update discriminator parameters
        disc_error_total = disc_error_real + disc_error_fake
        discriminator_opt.step()

        # train generator with fake images and calculate loss
        generator.zero_grad()
        label.fill_(real_label)

        # since the discriminator parameters are frozen, output of generator will be passed through discriminator
        output = discriminator(fake)
        gen_error = loss_function(output, label)

        # calculate gradients for generator and update generator parameters
        gen_error.backward()
        D_G_z2 = output.mean().item()
        generator_opt.step()
        
        # print the losses and save the real images and the generated images after every 200 steps
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader), disc_error_total.item(), gen_error.item(), D_x, D_G_z1, D_G_z2))
        
        if i % 200 == 0: # after every 200 iterations
            # save losses and the generated images
            gen_losses.append(gen_error.item())
            disc_losses.append(disc_error_total.item())
            vutils.save_image(real_cpu,'output/real_samples.png' ,normalize=True)
            fake = generator(fixed_noise)
            generated_fake = vutils.make_grid(fake)
            images.append(generated_fake)
            vutils.save_image(fake.detach(),'output/fake_samples_epoch_%03d.png' % (epoch), normalize=True)


# save the generated images as GIF file
imgs = [np.array(to_pil_image(img)) for img in images]
imageio.mimsave('output/generator_images.gif', imgs)

# plot and save the generator and discriminator loss
plt.figure()
plt.plot(gen_losses, label='Generator loss')
plt.plot(disc_losses, label='Discriminator Loss')
plt.legend()
plt.savefig('output/loss.png')