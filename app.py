import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import math

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(float(x)) for x in request.form.values()]
    final_features = np.asarray(int_features)
    input_data_reshaped = final_features.reshape(1,-1)
    prediction = model.predict(input_data_reshaped)
    
    return render_template('index.html',output_text="The predicted interest rate is approximately {}".format(prediction),
                          )
    
    
if __name__ == "__main__":
    app.debug=True
    app.run()