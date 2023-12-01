import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

st.set_page_config(layout="wide", page_title="Определение возраста")

st.write("## Определение возраста")
st.write(
    """Сетевой супермаркет Хлеб-Соль внедряет систему компьютерного зрения для обработки фотографий покупателей. 
    Фотофиксация в прикассовой зоне поможет определять возраст клиентов, чтобы: 
    - Анализировать покупки и предлагать товары, которые могут заинтересовать покупателей этой возрастной группы; 
    - Постройте модель, которая по фотографии определит приблизительный возраст человека."""
)
st.sidebar.header('Немного оинформации:')
st.sidebar.write('''После запуска вам будет необходимо загрузить пиксельную квадратную фотографию, 
                 в которую целиком помещается вся человеческая голова. 
                 Нейронная сеть лучше всего работает на изображениях, в которых люди смотрят прямо в камеру. Пример:
                 ''')
st.sidebar.image("https://github.com/tg-bomze/Face-Depixelizer/raw/master/example.jpg")
st.sidebar.write('''Обрезать фото вы можете 
                [ТУТ](https://www.iloveimg.com/crop-image)
                 ''')
                 
st.write("### Загрузите фотографию в формате: jpg, jpeg, png:")

uploaded_file = st.file_uploader("")

try: 
    if uploaded_file:
        loaded_model = tf.keras.saving.load_model("age_determination.h5")

        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

    
    
        prediction = loaded_model.predict(img_array)
        predicted_age = prediction[0][0]
        st.image(uploaded_file)
        st.write(predicted_age)
except:
    st.write('Загрузите файл необходимого расширения')