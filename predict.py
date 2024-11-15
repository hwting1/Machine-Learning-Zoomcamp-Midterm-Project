import json
import joblib
import streamlit as st

with open("columns_attribute.json", "r") as f_in:
    col_attrs = json.load(f_in)
pipeline = joblib.load("pipeline.joblib")

st.title("Food Delivery Time Prediction")

input_data = {}
for col_name, col_info in col_attrs.items():
    attr_type = col_info[0]

    if attr_type == "category":
        options = col_info[1]
        input_data[col_name] = st.selectbox(
            f"Select {col_name} (Category)", options
        )

    elif attr_type == "numeric":
        num_type = col_info[1]
        if num_type == "int64":
            input_data[col_name] = st.number_input(
                f"Enter {col_name}", step=1, format="%d"
            )
        elif num_type == "float64":
            input_data[col_name] = st.number_input(
                f"Enter {col_name}", step=0.1, format="%.1f"
            )

if st.button("Predict"):
    prediction = pipeline.predict([input_data])[0]
    st.markdown(
        f"<h2 style='text-align: center; color: blue;'>Estimated Delivery Time: {prediction:.2f} minutes</h2>",
        unsafe_allow_html=True
    )