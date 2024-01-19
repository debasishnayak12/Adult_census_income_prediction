from flask import Flask,render_template,request,jsonify,redirect
from src.INCOMEPREDICTION.pipeline.prediction_pipeline import PredictPipeline,custom_data
import numpy as np
app=Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict',methods=["GET","POST"])
def predict_data():
    if request.method == "POST":
        
        try:
            data=custom_data(
            
            age=float(request.form['age']),
            workclass =(request.form['workclass']),
            fnlwgt = float(request.form['fnlwgt']),
            education =request.form['education'],
            education_num = float(request.form['education_num']),
            marital_status =request.form['marital_status'],
            occupation = request.form['occupation'],
            relationship= request.form['relationship'],
            capital_gain = float(request.form['capital_gain']),
            hours_per_week = float(request.form['hours_per_week'])
        
            )
        
            final_data=data.get_data_as_dataframe()
            
            predict_pipeline=PredictPipeline()
            
            pred=predict_pipeline.predict(final_data)
            
            result=pred
        
            return render_template("result.html",final_result=result)
        
        except Exception as e:
            print(f"error :{str(e)}")
            return render_template("error.html", error_message=str(e))
    
    
if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)