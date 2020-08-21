# Handle table-like data and matrices
import pandas as pd
import numpy as np

# Modelling Algorithms
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pickle


# Modelling Helpers
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import LabelBinarizer, LabelEncoder


# Function to calculate missing values by column
def missing_values_table(df):
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
    
    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")
    
    # Return the dataframe with missing information
    return mis_val_table_ren_columns, mis_val_table_ren_columns.index

########## Fix detected missing values ############

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

############### Function to convert object type to interger type ############
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



# Read the data
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')
store = pd.read_csv('../data/store.csv')

train = pd.merge(train_df, store, how='inner', on='Store')
test = pd.merge(test_df, store, how='inner', on='Store')

# Drop all rows where Sales is 0
train = train[train['Sales'] > 0]
print(train.shape)

# remove all closed stores
train = train[train['Open'] != 0]

# specify the numeric columns we dont want to transform
# numeric_columns=numerical_columns
class myTransformer():
    """
    A pipeline class that fit the feature transform and make them ready for training and prediction
    Fix missing values using the function fix_missing()
    Encode Categorical Datatype to an Integer
    Extract Features from Date and generate more Features
    Normalized the data using Normalier()

    """
    def __init__(self):
        print('Initializing Binarizer......\n')
        # initialize all binarizer variables
        print('Binarizer Ready for Use!!!!!\n')
        
        # initialize the data scaler
        self.dataScaler=Normalizer()
        print('Scaler is Ready!!')
        
    # Fit all the binarisers on the training data
    def fit(self,input_data):
         # Check Missing
        print('Fixing Missing Values if any\n for int/float column fill with median otherwise No Data')
        table, missing_column = missing_values_table(input_data)
        fix_missing(df=input_data, column=missing_column)

        print('Features Ready for Fitting\n')
        
        input_data["StateHoliday"] = input_data["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})
        input_data['StateHoliday'] = input_data['StateHoliday'].astype(int)
        
        print('Features Encoded to Interger \n\nReady for Modelling\n')
        
    # Transform the input data using the fitted binarisers
    def transform(self, full_dataset,train=False):
        # making a copy of the input because we dont want to change the input in the main function
        input_data=full_dataset.copy()
        
        print('Extracting Features From Date object...\n')
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

        print('Data Transformed.... Up Next\n')
    
        # Drop Redundancy Columns
        print('Drop Unneccesary Columns..\n')
        columns_to_drop = [ 'Id', 'SalesPerCustomer', 'Customers', 'Date', 'PromoInterval',
                           'CompetitionOpenSinceYear','Promo2SinceYear', 'Store',
                           'dayofyear']
        for col in columns_to_drop:
            if col  in input_data.columns:
                input_data = input_data.drop(col, axis=1)

        # scale dataframe
        print('Scaling Data using Standard Scaler Method\n')
        table, missing_column = missing_values_table(input_data)
        fix_missing(df=input_data, column=missing_column)
        input_data = pd.DataFrame(self.dataScaler.fit_transform(input_data), columns=input_data.columns)
        print('Scaling Completed...')
        
        # this concatenates all the data
        print('Returning Data\n\n')
        print('Done!!!! Pipeline process completed')
        return input_data

######## Predict Sale #######
def predict_sales(data, model):
    if type(data) == dict:
        df = pd.DataFrame(data)
    else:
        df = data
    pipeline = myTransformer()
    pipeline.fit(df)
    prepared_df = pipeline.transform(df)
    prediction = model.predict(prepared_df)
    prediction = np.exp(prediction)
    print('prediction is ready')
    return prediction


############## Plot Forecast ###############

def plot_trend(data, prediction):
    import matplotlib.pyplot as plt
    data['prediction'] = pd.Series(prediction)
    date_sales = pd.concat([data['prediction'], data['Date']], axis=1)
    date_sales['Date'] = pd.to_datetime(date_sales['Date'])
    date_sales = date_sales.set_index('Date')
    fig, ax = plt.subplots(figsize=(18, 8))
    month_plot = date_sales.resample('D').sum()
    ax.plot(month_plot)
    ax.set_title('Trend of Predicted Sales by Day', size=25)
    plt.xticks(rotation=45, fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Average Sales per Day', fontsize=18)
    plt.show()
