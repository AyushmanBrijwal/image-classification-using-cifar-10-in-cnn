import streamlit as st
import tensorflow as tf
#from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import pickle

# Title and instructions
st.title("CIFAR-10 Image Classification")
st.write("Upload an image, and the model will classify it into one of the 10 CIFAR-10 categories.")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Load a pre-trained model
classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
models=['ANN','CNN']
#model=pickle.load(open('/Users/ashutoshpathak/Desktop/deepl/model_save','rb'))
#model2=pickle.load(open('/Users/ashutoshpathak/Desktop/deepl/model2_save','rb'))
model3=pickle.load(open('C:/Users/pikac/Desktop/image classification/model3_save','rb'))

#model = tf.keras.applications.ResNet50(weights='imagenet')

if uploaded_file is not None:
    # Display the uploaded image
    #st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.image(uploaded_file, caption='Uploaded Image', width=200)

    
    # Preprocess the image
    img = Image.open(uploaded_file)
    img = img.resize((32, 32))  # Resizing to match the model's input size
    img_array = np.array(img)
    #img_array=img_array//255
    img_array = np.expand_dims(img_array, axis=0)
    #img_array = preprocess_input(img_array)

            
    st.write(" Using CNN Model")
    # Make predictions
    #predictions = model2.predict(img_array)
    predictions = model3.predict(img_array)
    labe=classes[np.argmax(predictions)]
    #decoded_predictions = decode_predictions(predictions, top=3)[0]  # Get top 3 predictions

    st.subheader("Predictions:")
    
    #    st.write(f"{i + 1}: {label} ({score:.2f})")
    st.write(predictions)
    st.write(labe)


# Optional: Add a description or information about the model
myphoto=Image.open('iiitbh.jpg')

st.sidebar.image(myphoto)
st.sidebar.markdown("---")
st.sidebar.text('Minor Project')
st.sidebar.text('Under the guidance of Pradeep Kumar Biswal')

st.sidebar.markdown("---")
st.sidebar.subheader(" HELLO EVERYONE ")
st.sidebar.text("My Team Members are")
st.sidebar.text('Ashutosh kumar pathak  20001018  CSE')
st.sidebar.text('Gopal Pandey  2001010  CSE')
st.sidebar.text('Ayushman Singh  2001118  CSE')

st.sidebar.text('From IIIT BHAGALPUR')
#st.sidebar.markdown("[Email:-ashutosh.cse.20018@iiitbh.ac.in]")
#st.sidebar.markdown("[Link to GitHub:-https://github.com/ASHUTOSHPATHAK44]")
st.sidebar.markdown("---")
st.sidebar.markdown("---")
st.sidebar.subheader("About the Model")
st.sidebar.write("This model is trained on the CIFAR10 dataset and can be used to classify images into 10 categories.")
# Optional: Add a link to your GitHub repository or any additional information
st.sidebar.markdown("---")

# Optional: Add a footer

# Run the app
if __name__ == "__main__":
    st.markdown("---")
    st.write('These are the 10 classes of CIFAR10 dataset')
    st.write(classes)
    st.markdown("### Instructions")
    st.markdown("1. Upload an image.")
    st.markdown("2. The model will classify the image into one of the CIFAR-10 categories mentioned above.")
    