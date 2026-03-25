from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)
model = pickle.load(open('churn_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        tenure = float(request.form['tenure'])
        monthly = float(request.form['monthly_charges'])
        total = float(request.form['total_charges'])
        
       
        contract = int(request.form['contract']) # 0=Month-to-month, 1=One year, 2=Two year
        internet = int(request.form['internet']) # 0=DSL, 1=Fiber optic, 2=No
        
       
        input_data = np.zeros(19) 
        
        
        input_data[4] = tenure           # Column 5: Tenure
        input_data[7] = internet         # Column 8: InternetService
        input_data[14] = contract        # Column 15: Contract
        input_data[17] = monthly         # Column 18: MonthlyCharges
        input_data[18] = total           # Column 19: TotalCharges
        

        features_scaled = scaler.transform([input_data])
        
        prediction = model.predict(features_scaled)
        
        if prediction[0] == 1:
            result = "This customer is likely to LEAVE (Churn)."
            color = "danger"
        else:
            result = "This customer is likely to STAY (Loyal)."
            color = "success"
            
        return render_template('index.html', prediction_text=result, alert_class=color)
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}", alert_class="warning")

if __name__ == "__main__":
    app.run(debug=True)
