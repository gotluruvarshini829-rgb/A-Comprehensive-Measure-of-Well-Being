# app.py
import os
import pickle
import numpy as np
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression

# 1. SETUP ABSOLUTE PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'hdi_model.pkl')

# 2. AUTOMATICALLY CREATE TEMPLATES IF FLASK CAN'T FIND THEM
os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

home_html_path = os.path.join(TEMPLATE_DIR, 'home.html')
index_html_path = os.path.join(TEMPLATE_DIR, 'indexnew.html')

# Auto-create home.html if missing
if not os.path.exists(home_html_path):
    with open(home_html_path, 'w', encoding='utf-8') as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>HDI Prediction Hub</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 50px; background-color: #f4f6f9; color: #333; }
        .container { max-width: 800px; margin: auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #eee; padding-bottom: 20px; }
        .btn { background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold; }
        .btn:hover { background: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>Human Development Index (HDI) System</h2>
            <a href="/predict_page" class="btn">Go to Predictor</a>
        </div>
        <p style="margin-top:20px; line-height: 1.6;">
            The Human Development Index (HDI) is a summary measure of average achievement in key dimensions of human development. This application uses a machine learning Linear Regression model to estimate national metrics.
        </p>
    </div>
</body>
</html>""")

# Auto-create indexnew.html if missing
if not os.path.exists(index_html_path):
    with open(index_html_path, 'w', encoding='utf-8') as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>HDI Estimation Engine</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f6f9; }
        .container { max-width: 500px; margin: auto; background: white; padding: 25px; border-radius: 8px; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select { width: 100%; padding: 8px; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
        button { width: 100%; background: #28a745; color: white; padding: 10px; border: none; border-radius: 4px; font-size: 16px; cursor: pointer; }
        button:hover { background: #218838; }
        .result { margin-top: 20px; padding: 15px; background: #e2f0d9; border-left: 5px solid #28a745; font-size: 18px; font-weight: bold; }
        .back-link { display: inline-block; margin-top: 15px; color: #007bff; text-decoration: none; }
    </style>
</head>
<body>
    <div class="container">
        <h2>HDI Predictor Screen</h2>
        <form action="/predict" method="POST">
            <div class="form-group">
                <label>Select Target Analogue Country:</label>
                <select name="country">
                    {% for c in countries %}
                        <option value="{{ c }}">{{ c }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="form-group">
                <label>Life Expectancy (Years: 40 - 90):</label>
                <input type="number" step="0.1" name="life_exp" min="40" max="90" required value="75">
            </div>
            <div class="form-group">
                <label>Mean Years of Schooling (0 - 20):</label>
                <input type="number" step="0.1" name="schooling" min="0" max="20" required value="12">
            </div>
            <div class="form-group">
                <label>GNI Per Capita ($ USD):</label>
                <input type="number" name="gni" min="500" max="150000" required value="30000">
            </div>
            <button type="submit">Predict HDI Score</button>
        </form>
        {% if prediction_text %}
            <div class="result">{{ prediction_text }}</div>
        {% endif %}
        <a href="/" class="back-link">← Return to Homepage</a>
    </div>
</body>
</html>""")

# 3. AUTO-CREATE THE MODEL SEED FILE IF COPIES FAILED
if not os.path.exists(MODEL_PATH):
    print("--> [INFO] hdi_model.pkl not found. Seeding a fallback model automatically...")
    fallback_model = LinearRegression()
    # Dummy fitting data to make sure the object structure works immediately
    dummy_X = np.array([[75.0, 12.0, 30000.0], [60.0, 5.0, 2000.0], [84.0, 14.0, 65000.0]])
    dummy_y = np.array([0.750, 0.450, 0.960])
    fallback_model.fit(dummy_X, dummy_y)
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(fallback_model, file)

# 4. INITIALIZE FLASK ENGINE
app = Flask(__name__, template_folder=TEMPLATE_DIR)

with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_page')
def predict_page():
    countries = ["Switzerland", "Norway", "Iceland", "Hong Kong", "Australia", "Denmark", "Sweden"]
    return render_template('indexnew.html', countries=countries)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            life_exp = float(request.form['life_exp'])
            schooling = float(request.form['schooling'])
            gni = float(request.form['gni'])
            
            features = np.array([[life_exp, schooling, gni]])
            prediction = model.predict(features)[0]
            
            final_score = max(0.0, min(1.0, prediction))
            
            return render_template('indexnew.html', 
                                   prediction_text=f"Estimated Human Development Index (HDI): {final_score:.3f}",
                                   countries=["Switzerland", "Norway", "Iceland"])
        except Exception as e:
            return render_template('indexnew.html', 
                                   prediction_text="Error running metrics.",
                                   countries=["Switzerland", "Norway", "Iceland"])

if __name__ == '__main__':
    app.run(debug=True)