# Import nessesary files
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
from torchvision import transforms

import pandas as pd
from PIL import Image

import model
import image_preprocessor as ip

st.title("Digit Detector using MNIST Dataset")

# Run in GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.info(f"Device is running on {device}")

# Input the model path here
model = model.initialize_model(r'model/MNIST_Digit_Detector.pt')

st.success("""
Draw on the canvas, get the digits predicted with the confidence scores!
""")

st.header("Drawable Canvas")

# Create a canvas component
image_data = st_canvas(stroke_width=10,
                       stroke_color='#000000',
                       background_color='#FFFFFF',
                       width=700,
                       height=80
                       )

# Take the image data attribute and store it in the same variable.
# PS. Thats the only thing we need
image_data = image_data.image_data


# Do something interesting with the image data
if image_data is not None:

    digit_lis= ip.extract_digits(image_data)

    
    for snip in digit_lis:

        # Sometimes if we put a dot in the canvas or if the 
        # Canvas in empty on streamlit the NoneType Error is
        # generated. To check that this if condition is placed
        if snip is not None:
        
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

            with st.container():
                col1,col2 = st.columns(2)

                with col1:
                    st.image(invert,caption=f'Prediction: {digit}  |  Confidence: {conf:.2f}')

                with col2:
                    y_pred_prob_numpy = y_pred_prob.squeeze().to('cpu').numpy()

                    chart_data = pd.DataFrame(
                        y_pred_prob_numpy,
                        [0,1,2,3,4,5,6,7,8,9]                
                    )

                    chart_data.columns = ['Confidence']

                    st.bar_chart(chart_data)

        
        