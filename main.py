import pandas as pd
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
from test import test1
from home import test2
from flow import test3
from visualize import test4
st.set_page_config(page_title='Diabetes Prediction App', page_icon=':bar_chart:', layout='wide')
st.title('Diabetes prediction using data mining')

choice = option_menu(
        menu_title='Main Menu',
        options=['Home','Predict','Work flow','Visualize','Dataset'],
        menu_icon='cursor-fill',
        icons=['house','check2-square','diagram-2-fill','eye','file-earmark'],
        default_index=0,
        orientation = 'horizontal',
)

if choice == 'Home':
    test2()
if choice == 'Predict':
    test1()
if choice == 'Work flow':
    test3()
if choice == 'Visualize':
    test4()
if choice == 'Dataset':
    st.header('Dataset')
    df = pd.read_csv('diabetes.csv')
    df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
    def median_target(var):
        temp = df[df[var].notnull()]
        temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
        return temp
    columns = df.columns
    columns = columns.drop("Outcome")
    for i in columns:
        median_target(i)
        df.loc[(df['Outcome'] == 0) & (df[i].isnull()), i] = median_target(i)[i][0]
        df.loc[(df['Outcome'] == 1) & (df[i].isnull()), i] = median_target(i)[i][1]
    st.subheader('Features:')
    for i in df.columns:
        st.write("   -->",i)
    st.subheader('Description:')
    st.table(df.describe().T)
    st.subheader('Dataset:')
    st.dataframe(df)





