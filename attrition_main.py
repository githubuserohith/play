import attrition_model as am
import attrition_mlflow as al
import attrition_streamlit as ast
import os
import pandas as pd
import warnings
import streamlit as st

warnings.filterwarnings("ignore")

css = """
    body {
        background-image: url('Photo-compressed.png');
        background-size: cover;
    }
"""

# Set the custom CSS
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# os.chdir(r"")
df = pd.read_csv("IBM.csv")

model,pp,X_train,X_test,y_train,y_test,model_list = am.fn_model(df)
# print(model)
ast.fn_st(model,df,pp)
al.fn_mlflow(model,X_train,X_test,y_train,y_test,model_list)




