"""
نظام كشف الاحتيال البنكي - Flask Backend
"""
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# تحميل النموذج والأدوات عند بدء التشغيل
BASE = os.path.dirname(__file__)
model   = joblib.load(os.path.join(BASE, 'fraud_model.pkl'))
scaler  = joblib.load(os.path.join(BASE, 'scaler.pkl'))
le      = joblib.load(os.path.join(BASE, 'label_encoder.pkl'))
FEATURES = joblib.load(os.path.join(BASE, 'feature_names.pkl'))

MERCHANT_CATEGORIES = list(le.classes_)   # ['Clothing','Electronics','Food','Grocery','Travel']

@app.route('/')
def index():
    return render_template('index.html', categories=MERCHANT_CATEGORIES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # استخراج القيم
        amount           = float(data['amount'])
        transaction_hour = int(data['transaction_hour'])
        merchant         = data['merchant_category']
        foreign          = int(data['foreign_transaction'])
        loc_mismatch     = int(data['location_mismatch'])
        device_trust     = int(data['device_trust_score'])
        velocity         = int(data['velocity_last_24h'])
        age              = int(data['cardholder_age'])

        # ترميز نوع التاجر
        merchant_enc = int(le.transform([merchant])[0])

        # تجميع الميزات بنفس الترتيب
        features = pd.DataFrame([[amount, transaction_hour, merchant_enc,
                               foreign, loc_mismatch, device_trust,
                               velocity, age]])

        # تطبيع وتنبؤ
        features_scaled = scaler.transform(features)
        prediction      = int(model.predict(features_scaled)[0])
        probability     = float(model.predict_proba(features_scaled)[0][1])

        # حساب مستوى الخطر
        if probability < 0.3:
            risk_level = "منخفض"
            risk_color = "green"
        elif probability < 0.6:
            risk_level = "متوسط"
            risk_color = "orange"
        else:
            risk_level = "عالي"
            risk_color = "red"

        return jsonify({
            'success':     True,
            'prediction':  prediction,
            'probability': round(probability * 100, 1),
            'risk_level':  risk_level,
            'risk_color':  risk_color,
            'is_fraud':    prediction == 1,
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    print("=" * 50)
    print("  🚀 نظام كشف الاحتيال البنكي يعمل!")
    print("  🌐 افتح المتصفح على: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
