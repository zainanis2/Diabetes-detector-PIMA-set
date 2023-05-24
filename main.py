from flask import Flask, render_template, request
import joblib
import sklearn
import numpy as np



app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pregnancies = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        blood_pressure = int(request.form['blood_pressure'])
        skin_thickness = int(request.form['skin_thickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree = request.form['diabetes_pedigree']
        if len(diabetes_pedigree) == 0:
            diabetes_pedigree=0.0
        else:
            diabetes_pedigree=float(diabetes_pedigree)
        age = int(request.form['age'])
        result=""
        model = joblib.load('Diabeties_model.joblib')  # Update the filename to match your saved model
        data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]).reshape(1, -1)
        result = model.predict(data)
        if result==0:
            result="You are not likely to have diabetes"
        else:
            result = "You are likely to have diabetes"

        return render_template('result.html',result=result)

    return render_template('form.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)
