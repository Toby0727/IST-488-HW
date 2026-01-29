import streamlit as st

st.title("PDF Reader APP")
lab1 = st.Page('labs/lab1.py' , title = 'lab 1')
lab2 = st.Page('labs/lab2.py' , title = 'lab 2')
pg = st.navigation([lab2, lab1])
st.set_page_config(page_title= 'IST 488 labs')

