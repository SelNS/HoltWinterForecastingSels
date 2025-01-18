import pyrebase
import numpy as np
import pickle
import pandas as pd
from flask import Flask, flash, redirect,jsonify, render_template, request, send_file, make_response, session, abort, url_for, Response
from sensor import tds_sensor, pH_sensor, Temp_sensor, Hum_sensor
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import os
import json


app = Flask(__name__)       #Initialze flask constructor

#Add your own details
config = {
  "apiKey": "AIzaSyC2-w7RxNuhj1gMhaL45DPEpGo6Rr8uesY",
  "authDomain": "aeroforcast.firebaseapp.com",
  "databaseURL": "https://aeroforcast-default-rtdb.firebaseio.com",
  "storageBucket": "aeroforcast.appspot.com",
}

#initialize firebase
firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
db = firebase.database()

#Initialze person as dictionary
person = {"is_logged_in": False, "name": "", "email": "", "uid": ""}

#Login
@app.route("/")
def login():
    return render_template("login.html")

#Logout
@app.route("/logout")
def logout():
    return render_template("login.html")

#Welcome page
@app.route("/welcome")
def welcome():
    if person["is_logged_in"] == True:
        # Dapatkan waktu dan tanggal saat ini dari sistem
        current_datetime = datetime.now()
        current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

        # Konversi waktu dan tanggal saat ini ke format yang diinginkan
        input_date_time = current_datetime.strftime('%d/%m/%Y %H:%M:%S')

        # Konversi input_date_time ke format datetime
        desired_date_time = pd.to_datetime(input_date_time)

        # Forecast menggunakan model Holt-Winters untuk waktu tertentu
        modelTDS = pickle.load(open('hwsTDSforecast.pkl','rb'))
        desired_forecastTDS = round(modelTDS.forecast(steps=1).iloc[0])
        statusTDS=""
        if ((desired_forecastTDS >= 800) and (desired_forecastTDS <= 1200)):
            statusTDS="Optimal"
        if (desired_forecastTDS < 800):
            statusTDS="Low"
        if (desired_forecastTDS > 1200):
            statusTDS="High"

        # Forecast menggunakan model Holt-Winters untuk waktu tertentu
        modelpH = pickle.load(open('hwspHforecast.pkl','rb'))
        desired_forecastpH = round(modelpH.forecast(steps=1).iloc[0])
        statuspH=""
        if ((desired_forecastpH >= 5.5) and (desired_forecastpH <= 6.5)):
            statusTpH="Optimal"
        if (desired_forecastpH < 5.5):
            statuspH="Low"
        if (desired_forecastpH > 6.5):
            statuspH="High"

        # Forecast menggunakan model Holt-Winters untuk waktu tertentu
        modelTemp = pickle.load(open('hwsTempforecast.pkl','rb'))
        desired_forecastTemp = round(modelTemp.forecast(steps=1).iloc[0])
        statusTemp=""
        if ((desired_forecastTemp >= 18) and (desired_forecastTemp <= 29)):
            statusTemp="Optimal"
        if (desired_forecastTemp < 18):
            statusTemp="Low"
        if (desired_forecastTemp > 29):
            statusTemp="High"

        # Forecast menggunakan model Holt-Winters untuk waktu tertentu
        modelHum = pickle.load(open('hwsHumforecast.pkl','rb'))
        desired_forecastHum = round(modelHum.forecast(steps=1).iloc[0])
        statusHum=""
        if ((desired_forecastHum >= 60) and (desired_forecastHum <= 80)):
            statusHum="Optimal"
        if (desired_forecastHum < 60):
            statusHum="Low"
        if (desired_forecastHum > 80):
            statusHum="High"

        return render_template("welcome.html", email = person["email"], name = person["name"], current_time=current_time, predictionTDS = desired_forecastTDS, predictionpH = desired_forecastpH, predictionTemp = desired_forecastTemp, predictionHum = desired_forecastHum, satTDS=statusTDS, satpH=statuspH, satHum=statusHum, satTemp=statusTemp, date=input_date_time)
    else:
        return redirect(url_for('login'))

#If someone clicks on login, they are redirected to /result
@app.route("/result", methods = ["POST", "GET"])
def result():
    if request.method == "POST":        #Only if data has been posted
        result = request.form           #Get the data
        email = result["email"]
        password = result["pass"]
        try:
            #Try signing in the user with the given information
            user = auth.sign_in_with_email_and_password(email, password)
            #Insert the user data in the global person
            global person
            person["is_logged_in"] = True
            person["email"] = user["email"]
            person["uid"] = user["localId"]
            #Get the name of the user
            data = db.child("users").get()
            person["name"] = data.val()[person["uid"]]["name"]
            #Redirect to welcome page
            return redirect(url_for('welcome'))
        except:
            #If there is any error, redirect back to login
            return redirect(url_for('login'))
    else:
        if person["is_logged_in"] == True:
            return redirect(url_for('welcome'))
        else:
            return redirect(url_for('login'))

@app.route('/submit_dates', methods=['POST'])
def submit_dates():
    data = request.get_json()
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    return start_date, end_date

    # Simpan data ke file JSON
    with open('A:\\platihanPy\\aeros\\aeroforcast\\dates.json', 'w') as file:
        json.dump(data, file)

    return jsonify({"message": "Dates submitted successfully!"})

#Table Forcasting page
@app.route("/forcastTable")
def forcastTable():
    from datetime import datetime, timedelta
    import pandas as pd

    # Asumsikan 'fitted_temp', 'fitted_hum', 'fitted_ph', dan 'fitted_tds' adalah model Holt-Winters yang sudah dilatih
    modelTemp = pickle.load(open('hwsTempforecast.pkl','rb'))
    modelHum = pickle.load(open('hwsHumforecast.pkl','rb'))
    modelpH = pickle.load(open('hwspHforecast.pkl','rb'))
    modelTDS = pickle.load(open('hwsTDSforecast.pkl','rb'))

    # Dapatkan waktu dan tanggal saat ini dari sistem
    current_datetime = datetime.now()

    # Konversi waktu dan tanggal saat ini ke format yang diinginkan
    input_date_time = current_datetime.strftime('%d/%m/%Y %H:%M:%S')

    # Konversi input_date_time ke format datetime
    desired_date_time = pd.to_datetime(input_date_time)

    # Forecast menggunakan model Holt-Winters untuk 24 langkah ke depan (24 jam ke depan)
    desired_forecast_temp = modelTemp.forecast(steps=24)  # Prediksi 24 langkah ke depan untuk suhu
    desired_forecast_hum = modelHum.forecast(steps=24)    # Prediksi 24 langkah ke depan untuk kelembaban
    desired_forecast_ph = modelpH.forecast(steps=24)      # Prediksi 24 langkah ke depan untuk pH
    desired_forecast_tds = modelTDS.forecast(steps=24)    # Prediksi 24 langkah ke depan untuk TDS

    # Buat rentang waktu dengan interval 1 jam dari waktu saat ini
    future_dates = [desired_date_time + timedelta(hours=i) for i in range(24)]

    # Membuat DataFrame untuk hasil prediksi
    desired_forecast_data = pd.DataFrame({
        'Date': future_dates,
        'Temp_Prediction': round(desired_forecast_temp),
        'Hum_Prediction': round(desired_forecast_hum),
        'pH_Prediction': round(desired_forecast_ph),
        'TDS_Prediction': round(desired_forecast_tds)
    })

    # Konversi 'Date' ke format datetime dengan format yang benar
    desired_forecast_data['Date'] = pd.to_datetime(desired_forecast_data['Date']).replace("\n", "")

    # Tampilkan hasil prediksi
    return render_template("table.html", tables=[desired_forecast_data.to_html(classes='data', index=False)])


if __name__ == "__main__":
    app.run()