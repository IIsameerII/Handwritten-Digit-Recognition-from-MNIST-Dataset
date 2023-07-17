# Import nessesary files
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
from torchvision import transforms

import numpy as np
from PIL import Image
import cv2

import model
import image_preprocessor as ip

st.title("Digit Detector using MNIST Dataset")

# Input the model path here
model = model.initialize_model('model\MNIST_Digit_Detector.pt')

st.header("Drawable Canvas")
st.markdown("""
Draw on the canvas, get the digits predicted!
""")

# Specify brush parameters and drawing mode
b_width = st.slider("Brush width: ", 1, 100, 10)


# Create a canvas component
image_data = st_canvas(stroke_width=b_width,
                       stroke_color='#000000',
                       background_color='#FFFFFF'
                       )

# Take the image data attribute and store it in the same variable.
# PS. Thats the only thing we need
image_data = image_data.image_data

# Run in GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Do something interesting with the image data
if image_data is not None:

    digit_lis= ip.extract_digits(image_data)

    
    for snip in digit_lis:
        
        pil_image = Image.fromarray(snip)       
        
        # Transfrom the image to fit the model inputs
        transform = transforms.Compose([
            transforms.Resize(size=(28,28)),
            transforms.ToTensor(),
        ])

        # transform the image
        transformed_image = transform(pil_image).to(device)

        # Unsqueeze the tensor to [1,1,28,28]
        transformed_image = torch.unsqueeze(transformed_image,dim=0)

        with torch.inference_mode():
            # Set the model to evaluation mode
            model.eval()
            
            # Predit the digit of the model
            y_logits = model(transformed_image)
            y_pred_prob = torch.softmax(y_logits,dim=1)
            
            conf = torch.max(y_pred_prob).item()*100
            digit = str(torch.argmax(y_pred_prob).item())

        invert = ip.invert_colors_opencv(pil_image)
        st.image(invert,caption=f'Prediction: {digit}  |  Confidence: {conf:.2f}')

        
        