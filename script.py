# script.py
import pyrebase
import numpy as np
import pickle
import pandas as pd
from flask import Flask, flash, redirect,jsonify, render_template, request, send_file, make_response, session, abort, url_for, Response
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import os
import json
from flask import Flask, request

app = Flask(__name__)

def submit_dates(json_data=None):
    if json_data is None:
        json_data = request.get_json()
    # Lakukan sesuatu dengan json_data
    print(f"Data diterima: {json_data}")

@app.route('/submit', methods=['POST'])
def submit():
    submit_dates()
    return "Data submitted"

if __name__ == "__main__":
    with app.test_request_context(json={'key': 'value'}):
        submit_dates({"start_date": "15/08/2023 01:00:00",
    "end_date": "15/08/2023 23:00:00"})
