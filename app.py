from flask import Flask,render_template,request
import numpy as np
import pickle

app=Flask(__name__)
model=pickle.load(open('./models/lr_model.pkl','rb'))

@app.route("/")
def homepage():
    return render_template('homepage.html')


@app.route("/predict",methods=['POST'])
def predict():
    kms=float(request.form['kms'])
    car_age=float(request.form['car_age'])
    oprice=float(request.form['original_price'])

    fuel=request.form['fuel']
    if fuel=='Petrol':
        fuel=list([0,1])
    elif fuel=='Diesel':
        fuel=list([1,0])
    else:
        fuel=list([0,0])

    transmission=request.form['transmission']
    if transmission=='Manual':
        transmission=1
    else:
        transmission=0

    data=[np.array([kms,oprice,car_age,fuel[0],fuel[1],transmission])]
    result=model.predict(data)
    final_prediction=round(result[0],0)
    
    return render_template('homepage.html',prediction_value="Predicted car value is "+str(final_prediction))

    
    

if __name__=='__main__':
    app.run(debug=False) 