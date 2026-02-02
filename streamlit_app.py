import streamlit as st
HW 1 = st.Page('HW/HW 1.py' , title = 'HW 1')
HW 2 = st.Page('HW/HW 2.py' , title = 'HW 2', default=True)
pg = st.navigation([HW 1, HW 2])
st.set_page_config(page_title= 'HW Manager', initial_sidebar_state='expanded')


