# cv_project
Computer Vision multi-page streamlit app
- Multipage-приложение с использованием [streamlit](streamlit.io):
   - Генерация заданной цифры с помощью Conditional GAN (используем или на датасете MNIST, или то, на чем обучали модель во вторник)
   - Детекция объектов с помощью RCNN (из фреймворка Detectron2) или YOLO v5
      - Рекомендация: попробовать [YOLOv5](https://github.com/ultralytics/yolov5): просто посмотрите на этот [элегантный пример](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb)
   - Очищение документов от шумов с помощью автоэнкодера (данные берём из датасета [Denoising Dirty Documents](https://www.kaggle.com/c/denoising-dirty-documents/data))

