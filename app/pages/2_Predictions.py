import pickle
import streamlit as st
import pandas as pd

def header():
    st.header("WM Forecasting")

@st.cache_data
def load_data():
    df = pd.read_csv('app/artifactory/Processed_DatasetsAmount-of Waste-Generated-By-State 32121-0003.csv')
    return df

def states():
    data = load_data()
    s_df = data['States'].unique()
    s_df.tolist()

@st.cache_resource
def load_model():
    model = pickle.load(open('app/artifactory/wm-model.pkl','rb'))
    return model

def show_search_query():
    query = st.number_input("Enter Year  ",min_value=2022,max_value=2026,value=2022,step=1)
    waste = st.selectbox("Select Waste Type",('Residual household and bulky wastes',
       'Separately collected organic wastes',
       'Separately collected recyclables'))

    if query:
        df = predict(query,waste)
        df['Year'] = df['Year'].astype(int)
        st.write(df)

def predict(year,waste):

    data = load_data()
    model = load_model()
    states = data['States'].unique()

    df_input = pd.DataFrame(states,columns=['States'])

    df_input['Year'] = year

    df_input['Types of Waste'] = waste

    output = model.predict(df_input)

    df_predicted = pd.DataFrame(output, columns=['Total Household Waste Generated (Tons)','Household Waste Generated per Inhabitant (kg)'])

    df_final_bulk = pd.concat([df_input,df_predicted],axis=1)

    df_final_bulk.index = df_final_bulk.index+1

    return df_final_bulk


def main():
    header()
    show_search_query()


main()