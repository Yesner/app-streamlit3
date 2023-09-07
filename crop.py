import pandas as pd
import streamlit as st
from pycaret.classification import load_model, predict_model
from textblob import TextBlob
from PIL import Image

st.set_page_config(page_title="Crops")

st.header("Help Farmers Select the Best Crops")

st.markdown(""" Measuring essential soil metrics such as nitrogen, phosphorous, potassium levels, and pH value is an important aspect of assessing soil condition. 
            However, it can be an expensive and time-consuming process, which can cause farmers to prioritize which metrics to measure based on their budget constraints.

Farmers have various options when it comes to deciding which crop to plant each season. 
            Their primary objective is to maximize the yield of their crops, taking into account different factors. One crucial factor that affects crop growth is the condition of the soil in the field, which can be assessed by measuring basic elements such as nitrogen and potassium levels. Each crop has an ideal soil condition that ensures optimal growth and maximum yield.

""")

st.caption("Information about each field")

st.write('"N": Nitrogen content ratio in the soil.')
st.write('"P": Phosphorous content ratio in the soil.')
st.write('"K": Potassium content ratio in the soil.')
st.write('"pH" value of the soil.')


image = Image.open("farmer_in_a_field.jpg")

@st.cache(allow_output_mutation=True)
def get_model():
    return load_model('crop')

def predict(model, df):
    predictions = predict_model(model, data = df)
    return predictions['Label'][0]

def translation(texto):
    try:
        traducir = TextBlob(texto)
        traduccion = traducir.translate(from_lang='en', to = 'es')
        return str(traduccion)
    except:
        return str(texto)
    
model = get_model()

st.subheader("Do your prediction")

form = st.form("Crops")
N = form.slider('Nitrogen', min_value=0.0, max_value=140.0, value=0.0, step = 0.1, format = '%f')
P = form.slider('Phosphorous', min_value=5.0, max_value=145.0,value=0.0, step = 0.1, format = '%f')
K = form.slider('Potassium', min_value=5.0, max_value=205.0,value=0.0, step = 0.1, format = '%f')
ph = form.slider('pH', min_value=3.504752, max_value=9.935091,value=0.0, step = 0.1, format = '%f')

predict_button = form.form_submit_button('Predict')

input_dict = {'N' : N, 'P' : P,
              'K' : K, 'ph' : ph}

input_df = pd.DataFrame([input_dict])

if predict_button:
    out = predict(model, input_df)

    st.success(f'The predicted crop is: {translation(out)}.')
    st.image(image)