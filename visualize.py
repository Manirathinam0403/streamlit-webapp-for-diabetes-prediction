import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns

def test4():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.header('Visualization')
    choice = option_menu(
            menu_title='Options',
            options=['Histograms', 'Correlation Matrix', 'Pair Plot'],
            menu_icon='folder2-open',
            icons=['bar-chart-fill','grid-3x3','graph-up'],
            default_index=0,
            orientation="horizontal",
    )
    data = pd.read_csv('diabetes.csv')
    if choice == 'Histograms':
        st.header('Histograms')
        for feature in data.columns:
            data[feature].hist(bins=10,figsize=(10,10))
            plt.title(feature)
            plt.xlabel('Values')
            plt.ylabel('Frequency')
            st.pyplot()
    if choice == 'Correlation Matrix':
        st.header('Correlation Matrix')
        corr_mat = data.corr()
        plt.figure(figsize=(10,10))
        sns.heatmap(corr_mat, annot=True, cmap='RdYlGn')
        st.pyplot()

    if choice == 'Pair Plot':
        st.header('Pair plot')
        sns.pairplot(data=data, hue='Outcome')
        plt.show()
        st.pyplot()