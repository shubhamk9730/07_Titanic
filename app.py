from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle as pkl

model = pkl.load(open('Titanic_LO.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def pred():
    Pclass= request.form['Pclass']
    Gender= request.form['Gender']
    Age= request.form['Age']
    Parch= request.form['Parch']
    Fare= request.form['Fare']
    Embarked= request.form['Embarked']
    
    project_data = {"Gender":{'male':1, 'female':0}, 'columns' :[ 'Pclass','Gender','Age','Parch','Fare','Embarked']}

    arr = np.array([[Pclass, project_data['Gender'][Gender], Age, Parch, Fare, Embarked]])
    arr2= np.array(arr, dtype=float)

    pred = str(model.predict(arr2)[0])

    if pred == '0.0':
        return('Not Survived')
    else:
        return('Survived')

if __name__=='__main__':
    app.run(host='0.0.0.0', port=2020, debug=True)


