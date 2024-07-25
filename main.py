import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import streamlit as st
from PIL import Image, ImageOps
import numpy as np

# 定義類別標籤
class_labels = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# 重建模型架構
def build_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=2, activation='relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    return model

# 創建並加載權重
model = tf.keras.models.load_model('./fm_model.h5')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 定義圖像預處理和預測函數
def preprocess_and_predict(image):
    # 轉換圖像為灰度圖
    gray_image = ImageOps.grayscale(image)
    # 調整圖像大小為 28x28
    resized_image = gray_image.resize((28, 28))
    # 將圖像轉換為數組
    image_array = np.array(resized_image)
    # 正規化像素值
    image_array = image_array / 255.0
    # 添加批次維度
    image_array = np.expand_dims(image_array, axis=0)
    image_array = np.expand_dims(image_array, axis=-1)
    # 使用模型進行預測
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)
    return class_labels[predicted_class[0]], gray_image, resized_image

# Streamlit 應用
st.title('Fashion MNIST 圖像分類')
st.write('上傳一張圖片，模型將預測圖片的類型')

# 上傳圖片
uploaded_file = st.file_uploader("選擇一張圖片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='上傳的圖片', use_column_width=True)
    st.write("")

    # 圖像預處理和預測
    label, gray_image, resized_image = preprocess_and_predict(image)
    
    st.write("灰階圖像:")
    st.image(gray_image, caption='灰階圖像', use_column_width=True)
    
    st.write("調整大小的圖像 (28x28):")
    st.image(resized_image, caption='調整大小的圖像', use_column_width=False)

    st.write(f'預測結果: {label}')
