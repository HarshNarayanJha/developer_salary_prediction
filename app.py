import streamlit as st

from explore_page import show_explore_page
from predict_page import show_predict_page

# st.set_page_config(page_title="Software Developer Salary Prediction")
pages = ("Predict", "Explore")

page = st.sidebar.selectbox("Explore or Predict", pages)

if page == pages[0]:
    show_predict_page()
elif page == pages[1]:
    show_explore_page()
