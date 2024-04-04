import numpy as np
from flask import Flask,request,render_template
import pickle

app=Flask(__name__)

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    model=pickle.load(open("diabetes.pkl","rb"))
    float_features=[float(x) for x in request.form.values()]
    features=np.array(float_features)
    features=features.reshape(1,-1)
    prediction=model.predict(features)
    if prediction[0]==0:
        txt="You are not diabetic"
    else:
        txt="You are diabetic, please contact doctor"
    return render_template("index.html", prediction_text=txt)

@app.route("/predict1",methods=["POST"])
def predict1():
    model=pickle.load(open("kidney.pkl","rb"))
    float_features=[float(x) for x in request.form.values()]
    features=np.array(float_features)
    features=features.reshape(1,-1)
    prediction=model.predict(features)
    if prediction[0]==0:
        txt="You do not have any kidney disease."
    else:
        txt="You have kidney disease, please contact a doctor"
    return render_template("index.html", prediction_text=txt)

if __name__=="__main__":
    app.run(debug=True)