import streamlit as st
from PIL import Image
import urllib.request
import io
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms as T


import glob
from datetime import datetime
import os
import time


st.markdown("""<style>.main {background-color: #F5F5F5;}</style>""",unsafe_allow_html=True)
st.title("_:blue[YOlOv5 detection objects on Images]_")
st.header('''Read more [here](https://github.com/ultralytics/yolov5) ''')

urllib.request.urlretrieve('https://raw.githubusercontent.com/ultralytics/assets/main/yolov5/v70/splash.png', "splash.png")
start_image = Image.open("splash.png")

st.image(start_image, width=800)




def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
    return model



def image_input(source):
    
    if source == 'Upload my own image':
        image_file = st.file_uploader(label="Download image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded Image', use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data_yolo/uploads', str(ts)+image_file.name)
            outputpath = os.path.join('data_yolo/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            #call Model prediction--
            model = load_model() 
            pred = model(imgpath)
            pred.render()  # render bbox in image
            for im in pred.ims:
                im_base64 = Image.fromarray(im)
                im_base64.save(outputpath)

            #--Display predicton
            
            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='Model Prediction(s)', use_column_width='always')

    elif source == 'From test set': 
        # Image selector slider
        imgpath = glob.glob('data_yolo/images/*') ####!!!!!???
        imgsel = st.slider('Select random image from test set', min_value=1, max_value=len(imgpath), step=1) 
        image_file = imgpath[imgsel-1]
        submit = st.button("Predict!")
        col1, col2 = st.columns(2)
        with col1:
            img = Image.open(image_file)
            st.image(img, caption='Selected Image', use_column_width='always')
        with col2:            
            if image_file is not None and submit:
                #call Model prediction--
                model = load_model() 
                pred = model(image_file)
                pred.render()  # render bbox in image
                for im in pred.ims:
                    im_base64 = Image.fromarray(im)
                    im_base64.save(os.path.join('data_yolo/outputs', os.path.basename(image_file)))
                #--Display predicton
                    img_ = Image.open(os.path.join('data_yolo/outputs', os.path.basename(image_file)))
                    st.image(img_, caption='Model Prediction(s)')



def main():
    datasrc = st.radio("Choose input source:", ['From test set', 'Upload my own image'])
    return image_input(datasrc)
        
if __name__ == '__main__':
  
    main()
      


#####


# def load_image():
#     upload_file_img = st.file_uploader(label="Download image")
#     if upload_file_img:
#         img_data = upload_file_img.getvalue()
#         st.image(img_data)
#         img = Image.open(io.BytesIO(img_data))
#         return img

# model = load_model()
# with col1:
#     img = load_image()


# with col2:
#     st.write("")
#     st.write("")
#     st.write("")
#     st.write("")
#     res_button = st.button("detect objects on image")
#     if res_button:
#         results = model(img)
#         # results.render() 
#         results = results.show()
#         st.image(results)