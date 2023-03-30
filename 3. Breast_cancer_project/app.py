from flask import Flask, render_template,jsonify, redirect, request
import pickle
import numpy as np

model=pickle.load(open("BreastCancer.pkl", "rb"))
app=Flask(__name__)
@app.route("/")
def index():
   return render_template('main.html')

@app.route("/predict", methods =["POST"])
def predict():
    feature = [float(x) for x in request.form.values()]
    feature_final= np.array(feature).reshape(-1,30)
    prediction= model.predict(feature_final)

    if prediction=="M":
        return "<h1 style='color:green'> malignant </h1>"
    else:
        return "<h1 style='color:red'> benign </h1>"
    
if __name__=="__main__":
    app.run(debug=True)
    

