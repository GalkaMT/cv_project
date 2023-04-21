import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms as T
from PIL import Image


st.set_page_config(layout="wide")
st.header("Очистка документов от шумов 📝")

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # encoder 
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.Dropout(),
            nn.ReLU()
            )
        
        self.pool = nn.MaxPool2d(2, 2, return_indices=True, ceil_mode=True) #<<<<<< Bottleneck
        
        # decoder
        self.unpool = nn.MaxUnpool2d(2, 2)

        self.conv1_t = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU()
            )
        self.conv2_t = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=3),
            nn.LazyBatchNorm2d(),
            nn.Sigmoid()
            )     

    def encode(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x, indicies = self.pool(x) # ⟸ bottleneck
        return x, indicies

    def decode(self, x, indicies):
        x = self.unpool(x, indicies)
        x = self.conv1_t(x)
        x = self.conv2_t(x)
        return x

    def forward(self, x):
        latent, indicies = self.encode(x)
        out = self.decode(latent, indicies)      
        return out

@st.cache_resource    
def load_model():
    model = ConvAutoencoder()
    model.load_state_dict(torch.load('denoising_model_weights.pth'))
    model.eval()
    return model

model = load_model()

tab1, tab2 = st.tabs(["Демо", "Загрузить изображение"])

with tab1:
    st.subheader("Выберите изображение, чтобы посмотреть результат очистки:")
    demo_images = ['1', '2', '3']
    for i in range (len(demo_images)):
        demo_image = Image.open(f'denoising_demo/{i+1}.png')
        result = st.button(f'Пример очистки №{i+1}')
        if result:
            test_image = np.array(Image.open(f'denoising_demo/{i+1}.png').convert('L'))
            test_tensor = torch.Tensor(test_image).unsqueeze(0)
            test_tensor = (test_tensor.float()/255).unsqueeze(0)
            clean_test_image = model(test_tensor)
            col1, col2 = st.columns(2, gap='medium') # выводим оба изображения рядом
            with col1:
                st.write('Зашумлённое изображение:')
                fig, ax = plt.subplots(1,1)
                ax.imshow(test_image, cmap='gray')
                ax.axis('off')
                st.pyplot(fig)
            with col2:
                st.write('Очищенное изображение:')
                fig, ax = plt.subplots(1,1)
                ax.imshow(torch.permute(clean_test_image.squeeze(0).detach(), (1, 2, 0)), cmap='gray')
                ax.axis('off')
                st.pyplot(fig)
            
with tab2:
    st.subheader("Очистить собственное изображение")
    uploaded_file = st.file_uploader("Выберите черно-белое изображение в формате png, jpeg или jpg...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        user_image = np.array(Image.open(uploaded_file).convert('L'))
        image_tensor = torch.Tensor(user_image).unsqueeze(0)
        image_tensor = (image_tensor.float()/255).unsqueeze(0)
        clean_image = model(image_tensor)

        col1, col2 = st.columns(2, gap='medium') # выводим оба изображения рядом
        with col1:
            st.write('Загруженное изображение:')
            fig, ax = plt.subplots(1,1)
            ax.imshow(user_image, cmap='gray')
            ax.axis('off')
            st.pyplot(fig)
        with col2:
            st.write('Очищенное изображение:')
            fig, ax = plt.subplots(1,1)
            ax.imshow(torch.permute(clean_image.squeeze(0).detach(), (1, 2, 0)), cmap='gray')
            ax.axis('off')
            st.pyplot(fig)

        # кнопка загрузки очищенного изображения
        transform = T.ToPILImage()
        img = transform(clean_image.squeeze(0))
        col1, col2, col3 = st.columns(3) # перемещаем кнопку враво под очищенное изображение
        with col1:
            pass
        with col2:
            pass
        with col3 :
            img.save('generated_image.png')
            with open('generated_image.png', 'rb') as file:
                data = file.read()
            st.download_button(label='Загрузить очищенное изображение', data=data, file_name='generated_image.png', mime='image/png')  
