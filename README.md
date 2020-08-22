# RossManns Sales Prediction API
*A simple Webapp to forecast Sales of diffrent stores giving some features*

You work at ​ Rossmann Pharmaceuticals​  as a data scientist. The finance team wants to 
forecast sales in all their stores across several cities six weeks ahead of time. Managers in individual stores rely on their years of experience as well as their personal judgement to forecast sales.  
 
The data team identified factors such as promotions, competition, school and state holidays,
seasonality, and locality as necessary for predicting the sales across the various stores. 
Your job is to build and serve an end-to-end product that delivers this prediction to Analysts in the finance team.  

### Data
The data and features description for this challenge can be found [here](https://www.kaggle.com/c/rossmann-store-sales/data)

### Technology used
<ul>
    <li>Python</li>
    <li>Flask for web app</li> 
    <li>Heroku  for deployment</li> 
    <li>Jupyter notebook to generate graphs and charts</li>
</ul>

### File Structure

*templates*
    A boiler folder that contains static webpages and css designs <br>
*procfile* used by heroku 
*app.py* main python file to run prediction
*20-08-2020-16-32-31-xgboost.pkl* a serialized xgboost model to forecast sale. Model was evaluated with a validation dataset of 3 months and has a Root Mean Squared Error of 0.18 and a confidence interval of +/- 3.5
*requirement.txt*
*ml_model.py* A pipeline of transformations which will be used to transform all data before generating a prediction.
*Model Building.ipynb* A jupyter notebook version of *ml_model.py* More detailed and each cell is well commented and explained. Reason for Loss funtion is explained as well 
*RossMann Pharmaceutical Store Sales (Data Analysis)* Analysis of the stores data


**Referemces**
Reference 
[kaggle]​(​ https://www.kaggle.com/c/rossmann-store-sales/data)
[scikit-learn] (http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
 
 