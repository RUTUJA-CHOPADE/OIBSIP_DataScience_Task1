from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("iris_model.pkl")

# Species labels
species = ['Setosa', 'Versicolor', 'Virginica']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(data)[0]

        result = species[prediction]

        return render_template('index.html', prediction_text=f'Iris Species: {result}')
    except:
        return render_template('index.html', prediction_text="Invalid Input")

if __name__ == "__main__":
    app.run(debug=True)
