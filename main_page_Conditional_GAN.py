import torch 
import streamlit as st 
from torch import nn
import matplotlib.pyplot as plt


device = torch.device('cpu')
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
gen.load_state_dict(torch.load('cdcgan_weights.pth', map_location=device))

st.write('### Sudoku!')
base  = 3
side  = base*base

# pattern for a baseline valid solution
def pattern(r,c): return (base*(r%base)+r//base+c)%side

# randomize rows, columns and numbers (of valid base pattern)
from random import sample
def shuffle(s): return sample(s,len(s)) 
rBase = range(base) 
rows  = [ g*base + r for g in shuffle(rBase) for r in shuffle(rBase) ] 
cols  = [ g*base + c for g in shuffle(rBase) for c in shuffle(rBase) ]
nums  = shuffle(range(1,base*base+1))

# produce board using randomized baseline pattern
board = [ [nums[pattern(r,c)] for c in cols] for r in rows ]


squares = side*side
empties = squares * 3//4
for p in sample(range(squares),empties):
    board[p//side][p%side] = 0
n = -1
numSize = len(str(side))
fig, axs = plt.subplots(nrows=9, ncols=9, figsize = (2, 2))
for line in board:
    for number in line:
        n += 1
        ax = plt.subplot(9, 9, n + 1)
        if number == 0:
            ax.plot()
            plt.xticks([])
            plt.yticks([])
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
            ax.spines["left"].set_visible(True)
        else:
            random_part = torch.randn(1, 100, 1, 1).to(device)
            input_part = torch.nn.functional.one_hot(torch.tensor(number), 10).type(torch.float32).to(device)[None, :, None, None].to(device)

            fake_image = gen(random_part, input_part).detach().cpu().numpy().squeeze()
            ax.imshow(fake_image)
            ax.axis('off')
st.pyplot(fig)
st.write('### Если цифры шакальные, то вот подсказка:')
st.dataframe(data=board)

st.write('### Можно ввести любую последовательность цифр:')
numbers = st.text_input('Enter something', '23.06.1912')
numbers = numbers.replace('.', '')
if numbers is not None:

    fig, axs = plt.subplots(nrows=1, ncols=len(numbers), figsize = (2, 2))
    for number, ax in zip(numbers, axs.ravel()):

        random_part = torch.randn(1, 100, 1, 1).to(device)
        input_part = torch.nn.functional.one_hot(torch.tensor(int(number)), 10).type(torch.float32).to(device)[None, :, None, None].to(device)

        fake_image = gen(random_part, input_part).detach().cpu().numpy().squeeze()

        
        
        ax.imshow(fake_image, cmap = 'gray')
        ax.axis('off')

    st.pyplot(fig)

    
