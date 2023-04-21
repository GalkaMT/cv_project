import torch 
import streamlit as st 
from torch import nn
import matplotlib.pyplot as plt


device = torch.device('cpu')
stats = (.5),(.5)
latent_size = 100
ohe = 10

class Generator(nn.Module):
    def __init__(self):
        super().__init__()


        self.layer_y = nn.Sequential(
            
            #size = (batch_size, 10, 1, 1) 10 - 10D after one hot encoding of 10 classes
            nn.ConvTranspose2d(10, 128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )


        self.layer_z = nn.Sequential(

            #size = (batch_size, 100, 1, 1) 100 - 100D noise vector
            nn.ConvTranspose2d(100, 128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )


        self.layer_zy = nn.Sequential(

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride = 2, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride = 2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),    

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride = 2, padding=1, bias=False),
            nn.Tanh()          

        )
    
    def forward(self, input, labels):

        z = input.view(input.shape[0], input.shape[1], 1, 1)
        y = labels.view(labels.shape[0], labels.shape[1], 1, 1)

        z = self.layer_z(z)
        y = self.layer_y(y)

        conv = torch.cat((z, y), 1)

        out = self.layer_zy(conv)

        return out
    

gen = Generator()
gen.load_state_dict(torch.load('pages/cdcgan_weights.pth', map_location=device))

def denorm(img_tensor):
    return img_tensor*stats[1]+stats[0]

number = st.slider('Choose a number:', 0, 9, 1)

if number is not None:

    random_part = torch.randn(1, 100, 1, 1).to(device)
    input_part = torch.nn.functional.one_hot(torch.tensor(number), 10).type(torch.float32).to(device)[None, :, None, None].to(device)

    #fake_image = gen(random_part, input_part.detach().cpu().numpy().squeeze())

    fig, ax = plt.subplots(figsize = (1, 1))
    ax.imshow(gen(torch.randn(1, 100, 1, 1).to(device), torch.nn.functional.one_hot(torch.tensor(number), 10).type(torch.float32).to(device)[None, :, None, None].to(device)).detach().cpu().numpy().squeeze(), cmap = 'gray')
    ax.axis('off')

    st.pyplot(fig)

    #torch.randn(1, 100, 1, 1).to(device), torch.nn.functional.one_hot(torch.tensor(number), 10).type(torch.float32).to(device)[None, :, None, None].to(device)).detach().cpu().numpy().squeeze()