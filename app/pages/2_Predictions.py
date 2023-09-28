import pickle
import streamlit as st
import pandas as pd

model = pickle.load(open('app/artifactory/wm-model.pkl','rb'))

data = pd.read_csv('app/artifactory/Processed_DatasetsAmount-of Waste-Generated-By-State 32121-0003.csv')

states = data['States'].unique()

df_input = pd.DataFrame(states,columns=['States'])

df_input['Year'] = 2022

df_input['Types of Waste'] = 'Residual household and bulky wastes'

output = model.predict(df_input)

df_predicted = pd.DataFrame(output, columns=['Total Household Waste Generated (Tons)','Household Waste Generated per Inhabitant (kg)'])

df_final_bulk = pd.concat([df_input,df_predicted],axis=1)

df_final_bulk.index = df_final_bulk.index+1

st.write(df_final_bulk)