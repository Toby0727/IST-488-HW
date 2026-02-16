import streamlit as st
st.set_page_config(page_title= 'HW Manager', initial_sidebar_state='expanded')
HW_1 = st.Page('HW/HW 1.py' , title = 'HW 1')
HW_2 = st.Page('HW/HW 2.py' , title = 'HW 2')
HW_3 = st.Page('HW/HW 3.py' , title = 'HW 3')
HW_4 = st.Page('HW/HW 4.py' , title = 'HW 4', default=True)
pg = st.navigation([HW_1, HW_2, HW_3, HW_4])
pg.run()

