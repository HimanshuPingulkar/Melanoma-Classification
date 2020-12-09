import streamlit as st
import tensorflow as tf

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('my_model.hdf5')
    return model
model = load_model()
st.write("""
        # Melanoma Classification
""")

file = st.file_uploader("Please upload image file", type=["jpg", "png"])
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):
    size = (180, 180)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    
    return prediction

if file is None:
    st.text("Please upload image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    class_names = ['Benign', 'Malignant']
    string = "This image most likely belongs to {} with a {:.2f}% confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
    st.success(string)
