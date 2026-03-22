import streamlit as st
st.set_page_config(page_title= 'HW Manager', initial_sidebar_state='expanded')
HW_1 = st.Page('HW/HW 1.py' , title = 'HW 1')
HW_2 = st.Page('HW/HW 2.py' , title = 'HW 2')
HW_3 = st.Page('HW/HW 3.py' , title = 'HW 3')
HW_4 = st.Page('HW/HW 4.py' , title = 'HW 4', default=True)
HW_5 = st.Page('HW/HW 5.py' , title = 'HW 5')
HW_7 = st.Page('HW/HW 7.py' , title = 'HW 7')

pg = st.navigation([HW_1, HW_2, HW_3, HW_4, HW_5, HW_7])
pg.run()

