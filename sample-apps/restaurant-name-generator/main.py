import streamlit as st
import generate

st.title("Restaurant name")

cuisine = st.sidebar.selectbox(
    "Pick a cuisine", ("Indian", "Mexican", "Italian", "Japanese", "Thai", "Arabic"))

if cuisine:
    response = generate.generate_restaurant_name_and_item(cuisine)
    restaurant_name = response["restaurant_name"].strip()
    st.header(restaurant_name)
    menu_items = response["menu_items"].strip().split(",")
    st.write("**Menu Items**")

    for item in menu_items:
        st.write("-", item)
