import streamlit as st
from PIL import Image
import io
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms as T
from torchvision.models import inception_v3
from torchvision.models import Inception_V3_Weights


# def load_file():
#     uploaded_file = st.file_uploader( 'Upload your file', type=["jpg", "jpeg", "png"])
#     if uploaded_file is not None:
#         image_data = uploaded_file.getvalue()
#         st.image(image_data)
#         return Image.open(io.BytesIO(image_data))
#     else:
#         return None

def load_image():
    upload_file_img = st.file_uploader(label="Выберите изображение")
    if upload_file_img:
        img_data = upload_file_img.getvalue()
        st.image(img_data)
        img = Image.open(io.BytesIO(img_data))
        preprocess = T.Compose([
            T.Resize((299, 299)),
            T.ToTensor()])
        image_tensor = preprocess(img)
        return image_tensor

     
st.markdown(""""<style>.main {background-color: #F5F5F5;}</style>""",unsafe_allow_html=True)
st.title("Ultralytics YOLOv5 object detection")
st.write('''Read more [here](https://github.com/ultralytics/yolov5)''')

img = load_image()

res_button = st.button("Detect objects")
if res_button:
    # image transformation
    # trans = T.Compose([T.Resize((299,299)), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),T.ToTensor()])
    # prepared_img=trans(img)
    # Initialize model with the best available weights
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights)
    model.eval()
    # Initialize the inference transforms
    preprocess = weights.transforms()
    # apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    st.write(f"{category_name}: {100 * score:.1f}%")
