import streamlit as st 
import joblib 

st.title("Iris Flower Classification")

# Load the model that was created before
reloaded_model = joblib.load('iris_joblib')

# Sliders for input features 
sepal_length = st.slider("Sepal Length", 4.3, 7.9)
sepal_width = st.slider("Sepal Width", 2.0, 4.4)
petal_length = st.slider("Petal Length", 1.0, 6.9)
petal_width = st.slider("Petal Width", 0.1, 2.5)

# Predict the output category
if st.button('Predict'):
    y_pred = reloaded_model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    st.title(y_pred[0])
