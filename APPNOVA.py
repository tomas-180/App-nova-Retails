vimport json
import pickle
import datetime
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, Model, DateField, FloatField,
    CharField, IntegrityError  # <-- sku é string, usar CharField
)
from playhouse.shortcuts import model_to_dict
from sklearn.base import BaseEstimator, TransformerMixin
import os
from playhouse.db_url import connect
import joblib    
import logging
from functools import wraps
from threading import Lock
from collections import OrderedDict
from datetime import datetime, timedelta
from flask import make_response



app = Flask(__name__)  


# === =========================DATABASE ======================================

DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class PricePrediction(Model):
    sku = CharField()  # SKU é string
    time_key = DateField()
    pvp_is_competitorA = FloatField()
    pvp_is_competitorB = FloatField()
    pvp_is_competitorA_actual = FloatField(null=True)
    pvp_is_competitorB_actual = FloatField(null=True)

    class Meta:
        database = DB
        indexes = ((('sku', 'time_key'), True),)

DB.connect()
DB.create_tables([PricePrediction], safe=True)



# === Configura Logging ===

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('werkzeug').setLevel(logging.INFO)

db_lock = Lock()

# === Count Requests ===

# Contadores globais
forecast_total = 0
forecast_success = 0
forecast_fail = 0

actual_total = 0
actual_success = 0
actual_fail = 0

invalid_input_count = 0

# === Load Models and Data ===


with open('columns_novas.json') as f:
    columns = json.load(f)

with open('model_A_LightGBM.joblib', 'rb') as f:
    pipeline_A = joblib.load(f)

with open('model_B_LightGBM.joblib', 'rb') as f:
    pipeline_B = joblib.load(f)

with open("models/df_historico.pkl", "rb") as f:
    df_cache = pickle.load(f)
    df_cache['sku'] = df_cache['sku'].astype(str)

# === Helper Functions ===

# === Helper Functions ===
def validate_positive_price(price):
    return isinstance(price, (int, float)) and price >= 0

def validate_date_format(date_str):
    try:
        datetime.strptime(str(date_str), '%Y%m%d')
        return True
    except ValueError:
        return False

def date_not_before_oct_2024(date_str):
    try:
        date_obj = datetime.strptime(str(date_str), '%Y%m%d')
        cutoff = datetime(year=2024, month=10, day=1)
        return date_obj >= cutoff
    except Exception:
        return False

def validate_json_forecast(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        global invalid_input_count
        data = request.get_json()
        if not data:
            invalid_input_count += 1
            logger.warning("JSON body não enviado na request.")
            return jsonify({'error': 'JSON body required.'}), 422

        items = data if isinstance(data, list) else [data]

        for item in items:
            # Checa campos obrigatórios antes de tipo para evitar KeyError
            if 'sku' not in item or 'time_key' not in item:
                invalid_input_count += 1
                logger.warning(f"Faltando sku ou time_key: {item}")
                return jsonify({'error': 'sku and time_key are mandatory fields.'}), 422

            if not isinstance(item['sku'], str):
                invalid_input_count += 1
                logger.warning(f"Tipo inválido para sku: deve ser string")
                return jsonify({'error': 'Invalid type: sku must be string'}), 422

            if not isinstance(item['time_key'], int):
                invalid_input_count += 1
                logger.warning(f"Tipo inválido para time_key: deve ser integer")
                return jsonify({'error': 'Invalid type: time_key must be integer (yyyymmdd format without quotes)'}), 422

            time_key_str = str(item['time_key'])
            if len(time_key_str) != 8 or not time_key_str.isdigit():
                invalid_input_count += 1
                logger.warning(f"Formato inválido para time_key: {item['time_key']}")
                return jsonify({'error': 'time_key must be 8 digits (yyyymmdd format as integer)'}), 422
            
            if not validate_date_format(time_key_str):
                invalid_input_count += 1
                logger.warning(f"Formato inválido de time_key: {time_key_str}")
                return jsonify({'error': 'Invalid time_key format. Expected yyyymmdd.'}), 422

            if not date_not_before_oct_2024(time_key_str):
                invalid_input_count += 1
                logger.warning(f"time_key antes de 2024/10/01: {item['time_key']}")
                return jsonify({'error': 'time_key must be on or after 2024/10/01'}), 422

        return f(*args, **kwargs)
    return wrapper

def generate_features(df_cache, sku, time_key_str, modelo='A'):
    if sku not in df_cache['sku'].values:
        raise ValueError(f'SKU {sku} not found in Database.')
    
    # Conversão de data e extração de features temporais
    time_key = pd.to_datetime(str(time_key_str), format="%Y%m%d")
    year = time_key.year
    week = time_key.isocalendar().week
    month = time_key.month
    day_of_week = time_key.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0

    df_sku = df_cache[df_cache['sku'] == sku].copy()
    df_sku['time_key'] = pd.to_datetime(df_sku['time_key'])
    df_sku['week_of_year'] = df_sku['time_key'].dt.isocalendar().week
    df_sku['year'] = df_sku['time_key'].dt.year

    def get_mean_value(col):
        target_date = time_key
        week_of_year = target_date.isocalendar().week
        month = target_date.month

        historical_data = df_sku.copy()
        historical_data['time_key'] = pd.to_datetime(historical_data['time_key'], format="%Y%m%d")
        historical_data['week_of_year'] = historical_data['time_key'].dt.isocalendar().week
        historical_data['month'] = historical_data['time_key'].dt.month

        sku_data = historical_data[historical_data['sku'] == sku]

        # Tenta pela semana
        week_matches = sku_data[sku_data['week_of_year'] == week_of_year]
        if len(week_matches) >= 3:
            return week_matches[col].mean()

        # Tenta pelo mês
        month_matches = sku_data[sku_data['month'] == month]
        if len(month_matches) >= 3:
            return month_matches[col].mean()

        # Fallback geral
        if len(sku_data) >= 1:
            return sku_data[col].mean()

        # Último recurso: valor padrão
        return 0.0
    



    final_price_chain = get_mean_value('final_price_chain')
    discount_chain = get_mean_value('discount_chain')

    # Escolher o desconto consoante o modelo
    discount_comp = None
    if modelo == 'A':
        discount_comp = get_mean_value('discount_compA')
    elif modelo == 'B':
        discount_comp = get_mean_value('discount_compB')
    else:
        raise ValueError("Modelo inválido: deve ser 'A' ou 'B'")

    row_sample = df_sku.iloc[0]
    structure_level_2 = row_sample['structure_level_2']
    structure_level_3 = row_sample['structure_level_3']

    features = {
        'sku': sku,
        'structure_level_2': structure_level_2,
        'structure_level_3': structure_level_3,
        'final_price_chain': final_price_chain,
        'discount_chain': discount_chain,
        'day_of_week': day_of_week,
        'month': month,
        'is_weekend': is_weekend,
        'year': year,
        'week_of_year': week
    }

    if modelo == 'A':
        features['discount_compA'] = discount_comp
    elif modelo == 'B':
        features['discount_compB'] = discount_comp

    return pd.DataFrame([features])



# === Forecast Endpoint ===

forecast_requests_count = 0
actual_requests_count = 0

@app.route('/forecast_prices/', methods=['POST'])
@validate_json_forecast
def forecast_prices():

    global forecast_total, forecast_success, forecast_fail
    forecast_total += 1


    global forecast_requests_count
    forecast_requests_count += 1  # LOG: contar requests
    
    payload = request.get_json()
    sku = payload["sku"]
    time_key = payload["time_key"]
    time_key_dt = datetime.strptime(str(time_key), '%Y%m%d').date()
    
    logger.info(f"[forecast_prices] Request #{forecast_requests_count} - SKU: {sku}, time_key: {time_key}")

    with db_lock:
        try:
            PricePrediction.get(
                (PricePrediction.sku == sku) &
                (PricePrediction.time_key == time_key_dt)
            )
            forecast_fail += 1
            return jsonify({"error": "Forecast already exists for this sku and time_key"}), 422
        except PricePrediction.DoesNotExist:
            pass

        try:
            obs_df_A = generate_features(df_cache, sku, time_key, modelo='A')
            obs_df_B = generate_features(df_cache, sku, time_key, modelo='B')

            price_A = float(pipeline_A.predict(obs_df_A)[0])
            price_B = float(pipeline_B.predict(obs_df_B)[0])

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 422

        try:
            PricePrediction.create(
                sku=sku,
                time_key=time_key_dt,
                pvp_is_competitorA=price_A,
                pvp_is_competitorB=price_B,
            )
        except IntegrityError:
            return jsonify({"error": "Forecast already exists for this sku and time_key"}), 422
    forecast_success += 1
    logger.info(f"[forecast_prices] Predicted prices - competitorA: {price_A}, competitorB: {price_B}")

    return jsonify({
        "sku": sku,
        "time_key": time_key,
        "pvp_is_competitorA": price_A,
        "pvp_is_competitorB": price_B
    })



# === Actual Prices Endpoint ===

@app.route("/actual_prices/", methods=["POST"])
@validate_json_forecast
def actual_prices():

    global actual_total, actual_success, actual_fail
    actual_total += 1

    global actual_requests_count
    actual_requests_count += 1  # LOG: contar requests

    payload = request.get_json()
    sku = payload["sku"]
    time_key = payload["time_key"]
    time_key_dt = datetime.strptime(str(time_key), '%Y%m%d').date()

    # Log info dos preços reais recebidos
    pvp_compA_actual = payload.get("pvp_is_competitorA_actual")
    pvp_compB_actual = payload.get("pvp_is_competitorB_actual")
    logger.info(f"[actual_prices] Request #{actual_requests_count} - SKU: {sku}, time_key: {time_key}, "
                f"pvp_is_competitorA_actual: {pvp_compA_actual}, pvp_is_competitorB_actual: {pvp_compB_actual}")

    # Validação campos atual prices no payload
    for key in ["pvp_is_competitorA_actual", "pvp_is_competitorB_actual"]:
        if key not in payload:
            logger.warning(f"Missing field: {key}")
            return jsonify({"error": f"{key} is mandatory."}), 422
        if not isinstance(payload[key], (float, int)) or payload[key] < 0:
            logger.warning(f"Invalid value for {key}: {payload[key]}")
            return jsonify({"error": f"{key} must be a non-negative number."}), 422

    with db_lock:
        try:
            record = PricePrediction.get(
                (PricePrediction.sku == sku) & (PricePrediction.time_key == time_key_dt)
            )
        except PricePrediction.DoesNotExist:
            actual_fail += 1
            return jsonify({"error": "No forecast exists for this sku and time_key"}), 422

        record.pvp_is_competitorA_actual = float(payload["pvp_is_competitorA_actual"])
        record.pvp_is_competitorB_actual = float(payload["pvp_is_competitorB_actual"])
        record.save()


    actual_success += 1
    return jsonify(OrderedDict([
        ("sku", record.sku),
        ("time_key", time_key),
        ("pvp_is_competitorA", record.pvp_is_competitorA),
        ("pvp_is_competitorB", record.pvp_is_competitorB),
        ("pvp_is_competitorA_actual", record.pvp_is_competitorA_actual),
        ("pvp_is_competitorB_actual", record.pvp_is_competitorB_actual),
    ]))
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
