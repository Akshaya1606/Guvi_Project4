import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder
scaler = StandardScaler()
label_encoder = LabelEncoder()

ohe2 = OneHotEncoder(handle_unknown='ignore')

# Encode labels of multiple columns and store the encoders

with open("Selling.pkl","rb") as files:
    model1=pickle.load(files)
with open("Status.pkl","rb") as files:
    model2=pickle.load(files)
 
st.set_page_config(
        page_title="Industrial Copper",
        layout="wide",
    )

st.title("Industrial Copper Modeling")

tabs=st.tabs(['Selling Price Prediction','Status Prediction'])

with tabs[0]:
    st.header("Prediction of selling price using decision tree Regressor")
    st.write("**Please enter the following details to get the predicted selling price**")
    cus=st.number_input("Please enter the Customer ID")
    con=st.selectbox("Please select the Country code",(28,25,30,32,38,78,27,77,113,79,26,39,40,84,80, 107,89))
    qt=st.number_input("Please enter the Quantity in tons")
    it=st.selectbox("Please select the Item type",('W','S','PL','Others','WI','IPL','SLAWR'))
    app=st.selectbox("Please select Application",(10, 41, 28, 59, 15,4,38, 56,42,26,27,19,20,66,29,22,25,40,79,3,99,2,67,5,39,69,70,65,58,68))
    thic=st.number_input("Please enter the Thickness")
    width=st.number_input("Please enter the Width")
    pro=st.selectbox("Please enter the product Reference",(1670798778,1668701718,628377,640665,611993,1668701376,164141591,1671863738,1332077137,640405,1693867550, 1665572374, 1282007633, 1668701698,628117,1690738206,628112,640400,1671876026,164336407,164337175,1668701725,1665572032,611728,1721130331,1693867563,611733,1690738219,1722207579,929423819,1665584320, 1665584662, 1665584642))
    sta=st.selectbox("Please select the Status",('Won','Lost','Not lost for AM','Revised To be approved','Draft','Offered','Offerable','Wonderful'))
    button1=st.button("Predict the Selling Price")
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe2 = OneHotEncoder(handle_unknown='ignore')
    if button1:
        def preprocess_new_sample(sample, ohe, ohe2, scaler):
            sample_df = pd.DataFrame([sample])
            ity=np.array([['W','S','PL','Others','WI','IPL','SLAWR']])
            stt=np.array([['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM','Wonderful', 'Revised', 'Offered', 'Offerable']])
            ohe.fit(ity[[0]])
            sample_ohe = ohe.transform(ity[[0]]).toarray()
    
            ohe2.fit(stt[[0]])
            sample_be = ohe2.transform(stt[[0]]).toarray()
            sample_encoded = np.concatenate((sample_df[['quantity tons', 'application', 'thickness', 'width', 'country', 'customer', 'product_ref']].values, sample_ohe, sample_be), axis=1)
            # Scale the features
            sample_scaled = scaler.fit_transform(sample_encoded)
            return sample_scaled

        # Example of a new sample input
        new_sample = {
            'quantity tons': np.log(qt),
            'application': app,
            'thickness': np.log(thic),
            'width': width,
            'country': con,
            'customer': cus,
            'product_ref': pro,
            'status': sta,
            'item type': it
        }

        # Preprocess the new sample
        preprocessed_sample = preprocess_new_sample(new_sample, ohe, ohe2, scaler)

        # Predict the selling price for the new sample
        new_prediction = model1.predict(preprocessed_sample)
        ans=np.exp(new_prediction[0])
        st.success("The selling is predicted!!")
        st.write(f'**Predicted selling price for the new sample: {ans}**')

with tabs[1]:
    st.header("Prediction of selling price using Random Forest Classification")
    st.write("**Please enter the following details to get the predicted selling price**")
    'quantity tons','status','item type','application','thickness','width','country','customer','product_ref'
    cus1=st.number_input("Please enter the customer ID")
    con1=st.selectbox("Please select the country code",(28,25,30,32,38,78,27,77,113,79,26,39,40,84,80, 107,89))
    qt1=st.number_input("Please enter the quantity in tons")
    it1=st.selectbox("Please select the item type",('W','S','PL','Others','WI','IPL','SLAWR'))
    app1=st.selectbox("Please select application",(10, 41, 28, 59, 15,4,38, 56,42,26,27,19,20,66,29,22,25,40,79,3,99,2,67,5,39,69,70,65,58,68))
    thic1=st.number_input("Please enter the thickness")
    width1=st.number_input("Please enter the width")
    pro1=st.selectbox("Please enter the product reference",(1670798778,1668701718,628377,640665,611993,1668701376,164141591,1671863738,1332077137,640405,1693867550, 1665572374, 1282007633, 1668701698,628117,1690738206,628112,640400,1671876026,164336407,164337175,1668701725,1665572032,611728,1721130331,1693867563,611733,1690738219,1722207579,929423819,1665584320, 1665584662, 1665584642))
    sell1=st.number_input("Please enter the selling price")
    button2=st.button("Predict the Status")
    if button2:
        ohe = OneHotEncoder(handle_unknown='ignore')
        def preprocess_new_sample(sample,ohe, scaler):
            sample_df = pd.DataFrame([sample], columns=['country', 'quantity tons', 'item type', 'application', 'thickness', 'width','customer','product_ref','selling_price'])
            ity=np.array([['W','S','PL','Others','WI','IPL','SLAWR']])
            ohe.fit(ity[[0]])
            sample_ohe = ohe.transform(ity[[0]]).toarray()
            sample_encoded = np.concatenate((sample_df[['country', 'quantity tons', 'application', 'thickness', 'width','customer','product_ref','selling_price']].values, sample_ohe), axis=1)
            sample_scaled = scaler.fit_transform(sample_encoded)
            return sample_scaled
        # Example of a new sample input
        new_sample = {
            'country': con1,
            'quantity tons': np.log(qt1),
            'item type': it1,
            'application': app1,
            'thickness': np.log(thic1),
            'width': width1,
            'customer':cus1,
            'product_ref': pro1,
            'selling_price': sell1
        }

    # Preprocess the new sample
        preprocessed_sample = preprocess_new_sample(new_sample, ohe, scaler)
        y=['Lost','Won']
        label_encoder = LabelEncoder()
        y=label_encoder.fit_transform(y)
        # Predict the status for the new sample
        new_prediction = model2.predict(preprocessed_sample)

        # Decode the prediction
        new_prediction_decoded = label_encoder.inverse_transform(new_prediction)
        st.success("The status is predicted!!")

        st.write(f'**Predicted status for the new sample: {new_prediction_decoded[0]}**')


