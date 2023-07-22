import streamlit as st
import graphviz
def test3():
    st.header('Work flow')
    g = graphviz.Digraph()

    a1 = 'Start'
    a2 = 'Import Data from Dataset (diabetes.csv)'
    a3 = 'Prepare data for training'
    a4 = 'Feature/Attribute Extraction'
    a5 = 'Apply data mining models  (SVM, Logistic Regression, Naive Bayes, Random Forest)'
    a6 = 'Classification'
    a7 = 'Ready for deployment validation'
    a8 = 'End'
    lst = [a1,a2,a3,a4,a5,a6,a7,a8]
    for i in range(len(lst)-1):
        g.edge(lst[i],lst[i+1])

    st.graphviz_chart(g)