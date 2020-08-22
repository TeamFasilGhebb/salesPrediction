# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 19:00:39 2020

@author: kiiru
"""


import numpy as np
import pandas as pd

from flask import Flask, request, render_template
import pickle

from sklearn.preprocessing import Normalizer

app = Flask(__name__)
model = pickle.load(open('20-08-2020-16-32-31-00-xgboost.pkl', 'rb'))

store = pickle.load(open('store.pkl', 'rb'))

def missing_values_table(df):
    """
    calculate missing values in a dataframe df
    returns: missing values table that comprise of count % of missing and their datatype
    """
    # Total missing values
    mis_val = df.isnull().sum()
        
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
        
    # Datatypes of missing values
    mis_val_dtypes = df.dtypes
        
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtypes], axis=1)
        
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values', 2 : 'Data Types'})
        
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
    # Return the dataframe with missing information
    return mis_val_table_ren_columns, mis_val_table_ren_columns.index
    
def fix_missing(df, column):
    """
    The Function Fix missing values in the data (df) passed
    df = dataframe that contains the missing columns
    column = columns that has missing values
    """
    for col in column:
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna('No Data')
    return df

def encode_categorical_to_integer(data, columns):
    """
    convert or change onject datatype into integers
    for modelling
    Function takse 2 arguments 
    data : dataframe that contains column(s) of type object
    columns: a list of columns that are of type object
    the funtion does not return object, it does it computation implicitly
    """
    from sklearn.preprocessing import LabelEncoder
    for col in columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

class myTransformer():
    """
    A pipeline class that fit the feature transform and make them ready for training and prediction
    Fix missing values using the function fix_missing()
    Encode Categorical Datatype to an Integer
    Extract Features from Date and generate more Features
    Normalized the data using Normalier()
    """
    def __init__(self):
        # initialize the data scaler
        self.dataScaler=Normalizer()
        
    # Fit all the binarisers on the training data
    def fit(self,input_data):
         # Check Missing
        table, missing_column = missing_values_table(input_data)
        input_data = fix_missing(df=input_data, column=missing_column)
        
        input_data["StateHoliday"] = input_data["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})
        input_data['StateHoliday'] = input_data['StateHoliday'].astype(int)
                
    # Transform the input data using the fitted binarisers
    def transform(self,full_dataset,train=False):
        # making a copy of the input because we dont want to change the input in the main function
        input_data=full_dataset.copy()
        
        input_data['Date'] = pd.to_datetime(input_data['Date'])
        for attr in ['day', 'month', 'week', 'dayofweek', 'weekofyear']:
            input_data[attr] = getattr(input_data['Date'].dt, attr)
            input_data[attr] = getattr(input_data['Date'].dt, attr)
            
    ############################ New Features #################################
    
        input_data['year'] = input_data['Date'].dt.year
        input_data['is_weekend'] = (input_data['dayofweek'] >= 5)*1
        input_data['fortnight'] = input_data['day']%15
        input_data['which_fortnight'] = input_data['day']//15
        
        input_data['promo_per_competition_distance'] = input_data.groupby('Promo')['CompetitionDistance'].transform('mean')
        input_data['promo2_per_competition_distance'] = input_data.groupby('Promo2SinceWeek')['CompetitionDistance'].transform('mean')

        encode_categorical_to_integer(input_data, ['StoreType', 'Assortment'])         
        
        # Drop Redundancy Columns
        columns_to_drop = [ 'Id', 'SalesPerCustomer', 'Customers', 'Date', 'PromoInterval',
                           'CompetitionOpenSinceYear','Promo2SinceYear', 'Store',
                           'dayofyear']
        for col in columns_to_drop:
            if col  in input_data.columns:
                input_data = input_data.drop(col, axis=1)
        # scale dataframe
        table, missing_column = missing_values_table(input_data)
        input_data = fix_missing(df=input_data, column=missing_column)
        input_data = pd.DataFrame(self.dataScaler.fit_transform(input_data), columns=input_data.columns)
        
        # this concatenates all the data
        return input_data

def predict_sale(data, model):
    if type(data) == dict:
        df = pd.DataFrame(data)
    else:
        df = data
    pipeline = myTransformer()
    pipeline.fit(df)
    prepared_df = pipeline.transform(df)
    prediction = model.predict(prepared_df)
    prediction = np.exp(prediction)
    return prediction

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #eatures = [x for x in request.form.values()]
    Store = int(request.form.get("Store"))
    DayOfWeek = int(request.form.get("DayOfWeek"))
    Date = request.form.get("Date")
    Open = int(request.form.get("Open"))
    Promo = request.form.get("Promo")
    StateHoliday = request.form.get("StateHoliday")
    SchoolHoliday = request.form.get("SchoolHoliday")
    features = [[Store, DayOfWeek, Date, Open, Promo, StateHoliday, SchoolHoliday]]
    df = pd.DataFrame(data=features, columns=['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday'])
    df = pd.merge(df, store, on='Store')
    
    df = df.fillna(0)
    prediction = predict_sale(df, model)
    
    return render_template('index.html', prediction_text='$ {}'.format(prediction))

@app.route('/data', methods=['POST'])
def data():
    return render_template('roseman.html')

if __name__ == '__main__':
    app.run(debug=True)