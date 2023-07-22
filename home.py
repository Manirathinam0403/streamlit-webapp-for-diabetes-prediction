import streamlit as st
from streamlit_lottie import st_lottie
import requests
def test2():
    st.header('Home')

    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    lottie_url1 = "https://assets7.lottiefiles.com/packages/lf20_puciaact.json"
    lottie_gif1= load_lottieurl(lottie_url1)
    st_lottie(lottie_gif1, key="diabetes_gif1")

    lottie_url2 = "https://assets4.lottiefiles.com/packages/lf20_tbjuenb2.json"
    lottie_gif2 = load_lottieurl(lottie_url2)
    st_lottie(lottie_gif2, key="diabetes_gif2")

    fo = open('explanation.txt','r')
    content =  fo.read()
    st.write(content)