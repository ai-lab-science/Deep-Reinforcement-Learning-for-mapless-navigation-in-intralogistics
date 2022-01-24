import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import LeakyReLU
from utils import initialize_weights_he

# The MNIST datasets are hosted on yann.lecun.com that has moved under CloudFlare protection
# Run this script to enable the datasets download
# Reference: https://github.com/pytorch/vision/issues/1938
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
from Networks import ResNetBlock
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import cv2
from zu_resnet import ResNetEncoder


# define the NN architecture
class ConvAutoencoder_NAV2(nn.Module):
    def __init__(self, imgChannels=1, zDim=512,featureDim=12*10*10, fix_params=False):
        super(ConvAutoencoder_NAV2, self).__init__()
        self.featureDim = featureDim
        ## encoder layers ##
        #     https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
        self.encode = nn.Sequential(
                nn.Conv2d(imgChannels,  32, 5, padding=2)  ,
                nn.BatchNorm2d(32),
                nn.ReLU(),
                ResNetBlock(32,64,3), 
                ResNetBlock(64,128,3), 
                ResNetBlock(128,256,3), 
                ResNetBlock(256,128,3),  # 64x5x5 = 3200 feature vector
                ).apply(initialize_weights_he) 

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(128, 256,  2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128,  2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, imgChannels, 2, stride=2),
        ).apply(initialize_weights_he) 


    def fix_params(self):
        for param in self.encode.parameters():
            param.requires_grad = False
        for param in self.decode.parameters():
            param.requires_grad = False

    def encode_(self, x):
        return self.encode(x)

    def forward(self, x):
        x = self.encode(x)
        # print(x.shape)
        # x = x.reshape(64,5,5)
        x = self.decode(x)

        x = torch.sigmoid(x)        
        return x


# define the NN architecture
class ConvAutoencoder_NAV3(nn.Module):
    def __init__(self, imgChannels=1, zDim=512,featureDim=12*10*10, fix_params=False):
        super(ConvAutoencoder_NAV3, self).__init__()
        self.featureDim = featureDim
        ## encoder layers ##
        #     https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
        self.encode = ResNetEncoder(12,blocks_sizes=[64,128,256,384],deepths=[2,2,2,2])

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(384, 512,  2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256,  2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128,  2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, imgChannels,  2, stride=2)
        ).apply(initialize_weights_he) 


    def fix_params(self):
        for param in self.encode.parameters():
            param.requires_grad = False
        for param in self.decode.parameters():
            param.requires_grad = False

    def encode_(self, x):
        return self.encode(x)

    def forward(self, x):
        x = self.encode(x)
        # print(x.shape)
        # x = x.reshape(64,5,5)
        x = self.decode(x)

        x = torch.sigmoid(x)        
        return x


# define the NN architecture
class ConvAutoencoder_NAV4(nn.Module):
    def __init__(self, imgChannels=1, zDim=512,featureDim=12*10*10, fix_params=False):
        super(ConvAutoencoder_NAV4, self).__init__()
        self.featureDim = featureDim
        ## encoder layers ##
        #     https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
        self.encode = nn.Sequential(
                ResNetBlock(imgChannels,64,3), 
                ResNetBlock(64,128,3), 
                ResNetBlock(128,256,3), 
                ResNetBlock(256,128,3),  # 64x5x5 = 3200 feature vector
                ).apply(initialize_weights_he) 

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(128, 256,  2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128,  2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, imgChannels, 2, stride=2),
        ).apply(initialize_weights_he) 


    def fix_params(self):
        for param in self.encode.parameters():
            param.requires_grad = False
        for param in self.decode.parameters():
            param.requires_grad = False

    def encode_(self, x):
        return self.encode(x)

    def forward(self, x):
        x = self.encode(x)
        # print(x.shape)
        # x = x.reshape(64,5,5)
        x = self.decode(x)

        x = torch.sigmoid(x)        
        return x



# define the NN architecture
class ConvAutoencoder_NAV6(nn.Module):
    def __init__(self, imgChannels=1, zDim=1024,featureDim=64*5*5, fix_params=False):
        super(ConvAutoencoder_NAV6, self).__init__()
        self.featureDim = featureDim
        ## encoder layers ##
        #     https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
        self.encode = nn.Sequential(
                ResNetBlock(imgChannels,64,3), 
                ResNetBlock(64,128,3), 
                ResNetBlock(128,256,3), 
                ResNetBlock(256,64,3),  # 64x5x5 = 3200 feature vector,
                nn.Flatten(),
                nn.Linear(featureDim,zDim)
                ).apply(initialize_weights_he) 

        self. FC_1 = nn.Linear(zDim,featureDim)
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(64, 128,  2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 256,  2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            
        ).apply(initialize_weights_he) 


    def fix_params(self):
        for param in self.encode.parameters():
            param.requires_grad = False
        for param in self.decode.parameters():
            param.requires_grad = False

    def encode_(self, x):
        return self.encode(x)

    def forward(self, x):
        x = self.encode(x)

        x = x.view(-1, self.fedim)
        x = self.decode(x)

        x = torch.sigmoid(x)        
        return x


if __name__ == '__main__':


    GPU = True
    device_idx = 0
    if GPU:
        device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()


    channels = 3
    n_s_f = 4
    inputshape = (80,80,channels)
    cv2_resz = (80,80)
    imshape  = (channels,*cv2_resz)
    show_shape = (*cv2_resz,channels)


    model = ConvAutoencoder_NAV4(imgChannels=channels*n_s_f)
    # model.load_state_dict(torch.load("/home/developer/Training_results/Qricculum_Learning/big_and_small/final/Models/1/VAE_20"))
    model.load_state_dict(torch.load("/home/developer/Training_results/Qricculum_Learning/big_and_small/hoffentlich/VAE_80803_615"))
    model.eval()
    model.to(device)

    train_images = []
    test_images  = []

    moving_database = np.load("/home/developer/Training_results/Qricculum_Learning/big_and_small/hoffentlich/VAE_dtb_12_8080_final_hoffentlich.npy")
    # moving_database = np.load("/home/developer/VAE_dtb_12_128128_final.npy")
    # moving_database = np.load("/home/developer/Training_results/Qricculum_Learning/big_and_small/3/VAE_dtb_3_8080.npy")

    print(moving_database.shape)
    print(moving_database[0])

    stacked_images = []

    train_data =  (moving_database[0:45000]/ 2**8).astype(np.float32)
    test_data  =  (moving_database[45000:] / 2**8).astype(np.float32)

    print(train_data.shape)
    print(test_data.shape)

    # Create training and test dataloaders

    num_workers = 10
    # how many samples per batch to load
    batch_size = 32

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers,shuffle=True)

    import matplotlib.pyplot as plt


    infostring = "net: \n" + str(model) + " \n \n \n"
    
    print(infostring)
    filename = "/home/developer/Training_results/VA/"+"Infofile.txt"
    text_file = open(filename, "w")
    n = text_file.write(infostring)
    text_file.close()
    learning_rate = 0.01

    # specify loss function
    criterion = nn.MSELoss()

    # specify loss function
    # torch.optim.Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # from torch.optim.lr_scheduler import ExponentialLR
    from torch.optim.lr_scheduler import MultiStepLR

    # scheduler1 = ExponentialLR(optimizer, gamma=0.90)
    scheduler2 = MultiStepLR(optimizer, milestones=[30,50,70,90], gamma=0.25)

    # number of epochs to train the model
    n_epochs = 100

    # for epoch in range(1, n_epochs+1):
    #     # monitor training loss
    #     train_loss = 0.0
    #     test_loss  = 0.0
        
    #     ##################
    #     # train the model #
    #     ##################
    #     for data in train_loader:
    #         # _ stands in for labels, here
    #         # no need to flatten images
    #         images = data
    #         images = images.to(device)
    #         # clear the gradients of all optimized variables
    #         optimizer.zero_grad()
    #         # forward pass: compute predicted outputs by passing inputs to the model
    #         outputs = model(images).to(device)

    #         # output_decoder = decoder(images)
    #         # print(output_decoder)
    #         # print(output_decoder.shape)
    #         # calculate the loss
    #         loss = criterion(outputs, images)
    #         # backward pass: compute gradient of the loss with respect to model parameters
    #         loss.backward()
    #         # perform a single optimization step (parameter update)
    #         optimizer.step()
    #         # update running training loss
    #         train_loss += loss.item()*images.size(0)
                
    #     # print avg training statistics 
    #     train_loss = train_loss/len(train_loader)
    #     print('Epoch: {} \tTraining Loss: {:.6f}'.format(
    #         epoch, 
    #         train_loss
    #         ))
    #     for test_i_data in test_loader:
    #         # _ stands in for labels, here
    #         # no need to flatten images
    #         test_images = test_i_data
    #         test_images = test_images.to(device)
    #         # clear the gradients of all optimized variables
    #         with torch.no_grad():
    #             # forward pass: compute predicted outputs by passing inputs to the model
    #             outputs = model(test_images).to(device)

    #             loss = criterion(outputs, test_images)

    #             test_loss += loss.item()*test_images.size(0)
    #     print('Epoch: {} \tTesting Loss: {:.6f}'.format(
    #         epoch, 
    #         test_loss
    #         ))
    #     torch.save(model.state_dict(), "/home/developer/Training_results/VA/VAE_RESNET18"+str(epoch))
    #     # scheduler1.step()
    #     scheduler2.step()


    # obtain one batch of test images
    dataiter = iter(test_loader)
    while True: 
        show_images = dataiter.next()
        show_images = show_images.to(device)
        # get sample outputs
        output = model(show_images)
        # prep images for display
        show_images = show_images.detach().cpu().numpy()

        # output is resized into a batch of iages
        output = output.view(batch_size,n_s_f*channels,*cv2_resz)
        # use detach when it's an output that requires_grad
        output = output.detach().cpu().numpy()

        print(output.shape)
        print(show_images.shape)

        # torch.save(model.state_dict(), "/home/developer/Training_results/VAE")

        # plot the first ten input images and then reconstructed images
        fig, axes = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(20,20))

        axes[0][0].imshow(show_images[0][0:3].reshape(show_shape))
        axes[0][0].get_xaxis().set_visible(False)
        axes[0][0].get_yaxis().set_visible(False)
        axes[0][1].imshow(show_images[0][3:6].reshape(show_shape))
        axes[0][1].get_xaxis().set_visible(False)
        axes[0][1].get_yaxis().set_visible(False)
        axes[0][2].imshow(show_images[0][6:9].reshape(show_shape))
        axes[0][2].get_xaxis().set_visible(False)
        axes[0][2].get_yaxis().set_visible(False)
        axes[0][3].imshow(show_images[0][9:12].reshape(show_shape))
        axes[0][3].get_xaxis().set_visible(False)
        axes[0][3].get_yaxis().set_visible(False)


        axes[1][0].imshow(output[0][0:3].reshape(show_shape))
        axes[1][0].get_xaxis().set_visible(False)
        axes[1][0].get_yaxis().set_visible(False)
        axes[1][1].imshow(output[0][3:6].reshape(show_shape))
        axes[1][1].get_xaxis().set_visible(False)
        axes[1][1].get_yaxis().set_visible(False)
        axes[1][2].imshow(output[0][6:9].reshape(show_shape))
        axes[1][2].get_xaxis().set_visible(False)
        axes[1][2].get_yaxis().set_visible(False)
        axes[1][3].imshow(output[0][9:12].reshape(show_shape))
        axes[1][3].get_xaxis().set_visible(False)
        axes[1][3].get_yaxis().set_visible(False)

        # input images on top row, reconstructions on bottom
        # for show_images, row in zip([show_images, output], axes):
        #     for img, ax in zip(show_images, row):
        #         ax.imshow(img[0:3].reshape(show_shape))
        #         ax.get_xaxis().set_visible(False)
        #         ax.get_yaxis().set_visible(False)

        plt.show()