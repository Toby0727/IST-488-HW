import streamlit as st

st.title("PDF Reader APP")
lab1 = st.Page('labs/lab1.py' , title = 'lab 1')
lab2 = st.Page('labs/lab2.py' , title = 'lab 2', default=True)
pg = st.navigation([lab1, lab2])
st.set_page_config(page_title= 'PDF READER APP')
pg.run()

