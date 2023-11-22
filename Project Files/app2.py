# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 10:02:27 2023

@author: Raj Kumar
"""

import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request
import pickle
import datetime as dt
import calendar
import os
 

app = Flask(__name__)

loaded_model = pickle.load(open('rf_model.pkl','rb'))
fet = pd.read_csv('merged_data.csv')

feature_importance_dict = {
    'dept': 0.2,
    'size': 0.15,
    'store': 0.12,
    'Week': 0.1,
    'CPI': 0.08,
    'temp': 0.07,
    'b': 0.06,
    'Month': 0.05,
    'a': 0.04,
    'IsHoliday': 0.03,
    'Year': 0.02,
    'c': 0.01
}
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    store = request.form.get('store')
    dept = request.form.get('dept')
    size=request.form.get('size')
    temp=request.form.get('temp')
    a=request.form['aRadio']
    b=request.form['bRadio']
    c=request.form['cRadio']
    CPI=request.form.get('CPI')
    Unemployment=request.form.get('Unemployment')
    IsHoliday = request.form['isHolidayRadio']    
    date = request.form.get('date')
    d=dt.datetime.strptime(date, '%Y-%m-%d')
    Month = d.month
    Year = (d.year)
    Week=d.isocalendar().week
    month_name=calendar.month_name[Month]
    print("year = ", type(Year))
    print("year val = ", Year, type(Year), Month)
    X_test = pd.DataFrame({'Store': [store], 'Dept': [dept],'IsHoliday':[IsHoliday], 
                           'Temperature':[temp], 'CPI':[CPI],'Size':[size], 
                           'Month':[Month], 'Year':[Year],'Week':[Week],
                           'A':[a], 'B':[b], 'C':[c]})
    print("X_test = ", X_test.head())
    print("type of X_test = ", type(X_test))
    print("predict = ", store, dept, date, IsHoliday)

    y_pred = loaded_model.predict(X_test)
    output=round(y_pred[0],2)
    print("predicted = ", output)
    insights = generate_insights(output)
    
    most_significant_feature = max(feature_importance_dict,key=feature_importance_dict.get)
    print("significant_feature=",most_significant_feature)
    return render_template('index.html', output=output, store=store, dept=dept, 
                           month_name=month_name, Year=Year, insights=insights,feature_importance=feature_importance_dict, 
                           most_significant_feature=most_significant_feature)

def generate_insights(output):
    insights = "No specific insights available."
    if output > 50000:
        insights = "The predicted sales are high. Consider potential factors contributing to this peak."
    elif output < 5000:
        insights = "The predicted sales are relatively low. Evaluate potential reasons for the decrease."
    else:
        insights = "The predicted sales are within a moderate range."

    return insights
    

if __name__ == "__main__":
    app.run(debug=False)



