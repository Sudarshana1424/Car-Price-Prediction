import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
import joblib 
from joblib import dump


import datetime

import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score





st.header('Used Car Price :red[Prediction...] :oncoming_automobile:', divider='rainbow')

st.write("\n\n"*2)


with st.sidebar:
    st.subheader('Car Specs to Predict Price')

make_model = st.sidebar.selectbox("Car Company: :red_car:", ('Maruti', 'Hyundai', 'Honda', 'Audi', 'Nissan', 'Toyota', 'Volkswagen', 'Tata', 'Land', 'Mitsubishi', 'Renault', 'Mercedes-Benz', 'Bmw', 'Mahindra', 'Ford', 'Porsche', 'Datsun', 'Jaguar', 'Volvo', 'Chevrolet', 'Skoda', 'Mini', 'Fiat', 'Jeep', 'Smart', 'Ambassador', 'Isuzu', 'Force', 'Bentley', 'Lamborghini'),key=1)
year = st.sidebar.number_input("Year: :calendar:",min_value=1000, max_value=3000, value=1900, step=100,key=2)
KM_Driven = st.sidebar.number_input("KM Driven: :running:",min_value=0, max_value=775000, value=10000, step=2000,key=3)
fuel_Type = st.sidebar.selectbox("Fuel Type: :fuelpump:", ('CNG', 'Diesel', 'Petrol', 'LPG', 'Electric'),key=4)
owner_Type = st.sidebar.selectbox("OwnerShip: :man_in_business_suit_levitating:", ('First', 'Third', 'Second', 'Fourth & Above'),key=5)
seats = st.sidebar.number_input("No. Of Seats: :seat:",min_value=0, max_value=15, value=2, step=2,key=6)
mileage = st.sidebar.number_input("Mileage in Km's: :chart_with_upwards_trend:",min_value=0, max_value=90, value=5, step=5,key=7)
engine = st.sidebar.number_input("Engine CC: :gear:",min_value=100, max_value=10000, value=500, step=500,key=8)
# mileage = st.sidebar.number_input("Mileage in Km's:",min_value=0, max_value=90, value=5, step=5,key=9)
hp_kW = st.sidebar.number_input("Horse Power in bhp: :horse:",min_value=10, max_value=1000, value=30, step=50,key=10)
transmission_Type = st.sidebar.radio("Transmission Type :motorized_wheelchair:", ('Manual', 'Automatic'),key=11)


my_dict = {"manufacturer":make_model, "year":year, "Km":KM_Driven, "fuel_type":fuel_Type, "owner_type":owner_Type,'seats':seats,'mileage':mileage,'engine':engine, 'hp_kw':hp_kW,'trans_type':transmission_Type}
df = pd.DataFrame.from_dict([my_dict])

cols = {
    "make_model": "Car Manufacturer",
    "year": "Year",
     "Km": "km Traveled",
    "fuel_type":"Fuel Type", "owner_type":'Owner Type','seats':"No of Seats",'mileage':"Mileage",'engine':"Engine", 'hp_kw':"Power",'trans_type':"Transmission Type"

}

df_show = df.copy()
df_show.rename(columns = cols, inplace = True)
st.write("Selected Specs: \n")
st.table(df_show)



if st.button("Predict"):
        #------------------->Preprocessing and Model Training.
    dataset = pd.read_csv("data/dataset.csv")
    dataset_copy=dataset.copy()
    dataset.head(5)

    dataset=dataset[['Name', 'Location', 'Year', 'Kilometers_Driven',
           'Fuel_Type', 'Transmission', 'Owner_Type', 'Mileage', 'Engine', 'Power',
           'Seats', 'New_Price', 'Price']]

    dataset['Manufacturer']=[i.split(" ")[0].lower() for i in dataset['Name']]
    dataset['Mileage_Num'] = dataset['Mileage'].str.extract(r'(\d+\.\d+|\d+)', expand=True)
    dataset['Mileage_Num']=dataset['Mileage_Num'].astype(float)
    dataset["Mileage_Num"].fillna(dataset['Mileage_Num'].mode()[0],inplace=True)
    dataset['Engine_Num'] = dataset['Engine'].str.extract(r'(\d+\.\d+|\d+)', expand=True)
    dataset['Engine_Num']=dataset['Engine_Num'].astype(float)
    dataset["Engine_Num"].fillna(dataset['Engine_Num'].mode()[0],inplace=True)
    dataset['Power_Num'] = dataset['Power'].str.extract(r'(\d+\.\d+|\d+)', expand=True)
    dataset['Power_Num']=dataset['Power_Num'].astype(float)
    dataset["Power_Num"].fillna(dataset['Power_Num'].mode()[0],inplace=True)

    dataset["Seats"].fillna(dataset['Seats'].mode()[0],inplace=True)

    def create_dict(val_list,start_value):
        temp_dict=dict()
        val_list=set(val_list)
        for i in val_list:
            temp_dict[i]=start_value
            start_value+=1
        rev_temp_dict=dict(zip(temp_dict.values(),temp_dict.keys()))
        return temp_dict,rev_temp_dict   

    manufacturer_dict,rev_manufacturer_dict=create_dict(dataset['Manufacturer'].unique(),0)
    fuel_dict,rev_fuel_dict=create_dict(dataset['Fuel_Type'].unique(),0)
    transmission_dict,rev_transmission_dict=create_dict(dataset['Transmission'].unique(),0) 
    Owner_Type_dict,rev_Owner_Type_dict=create_dict(dataset['Owner_Type'].unique(),0) 


    dataset_copy2=dataset.copy()

    dataset=dataset[['Year', 'Kilometers_Driven', 'Fuel_Type',
           'Transmission', 'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats', 'Manufacturer', 'Mileage_Num', 'Engine_Num',
           'Power_Num','Price']]
    X=dataset[['Year', 'Kilometers_Driven', 'Fuel_Type',
           'Transmission', 'Owner_Type',  'Seats', 'Manufacturer', 'Mileage_Num', 'Engine_Num',
           'Power_Num']]
    cols=['Manufacturer','Fuel_Type','Transmission','Owner_Type']
    dicts=[manufacturer_dict,fuel_dict,transmission_dict,Owner_Type_dict]
    for i in range(len(cols)):
        X[cols[i]]=X[cols[i]].replace(dicts[i])

    y=dataset['Price']

    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                        test_size = 0.3,
                                                        random_state = 24)
    standardScaler = StandardScaler()
    standardScaler.fit(X_train)
    X_train = standardScaler.transform(X_train)
    X_test = standardScaler.transform(X_test)

    linearRegression = LinearRegression()
    linearRegression.fit(X_train, y_train)
    y_pred = linearRegression.predict(X_test)
    print(r2_score(y_test, y_pred))

    rf = RandomForestRegressor(n_estimators = 100)
    rf.fit(X_train, y_train)

    #---------------------->
    pred_values=[year,KM_Driven,fuel_dict[fuel_Type],transmission_dict[transmission_Type],Owner_Type_dict[owner_Type],seats,
    manufacturer_dict[make_model.lower()],mileage,engine,hp_kW]
    sampl=standardScaler.transform([pred_values])
    predicted_price=rf.predict(sampl)
#     col1, col2 = st.columns(2)
    st.write(F"The estimated value of car price is :money_with_wings: ₹ {round(predicted_price[0],2)} L")
#     col1.write("The estimated value of car price is €")
#     col2.write(predicted_price[0].astype(int))


