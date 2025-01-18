import pandas as pd
import numpy as np
import io
import os
import json
import pickle
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from sklearn.metrics import mean_squared_error as MSE

#import data perangkat sensor dari firebase
import firebase_admin
from firebase_admin import credentials, db

# Replace 'path/to/serviceAccountKey.json' with the path to your Firebase service account key JSON file
cred = credentials.Certificate('A:\\platihanPy\\aeros\\aeroforcast\\ServiceAccount.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://aeroforcast-default-rtdb.firebaseio.com/'
})

# Now you can access the Firebase Realtime Database
ref = db.reference('datacollect/')  # The reference should be relative to the database root
data = ref.get()

aero = pd.DataFrame(data)
# Finding the shape of the dataframe
print(aero.shape)
# Simpan sebagai file CSV
aero.to_csv('A:\\platihanPy\\aeros\\aeroforcast\\Sensordata.csv', index=False)
# Baca file JSON
with open('A:\\platihanPy\\aeros\\aeroforcast\\dates.json', 'r') as file:
    json_data = json.load(file)

# Ekstrak bagian 'data' dari JSON
datastart = json_data['start_date']
dataend = json_data['end_date']

# Buat DataFrame dari tanggal
timedate = pd.DataFrame({
    'start_date': datastart,
    'end_date': dataend
}, index=[0])
# Convert to datetime
timedate['start_date'] = pd.to_datetime(timedate['start_date'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
timedate['end_date'] = pd.to_datetime(timedate['end_date'], format='%d/%m/%Y %H:%M:%S', errors='coerce')

print("DataFrame dan timedate telah disimpan sebagai file 'A:\\platihanPy\\aeros\\aeroforcast\\Sensordata.csv'.")

def tds_sensor():
    import pandas as pd
    import numpy as np
    import io
    import os
    import json
    import pickle
    from matplotlib import pyplot as plt
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
    from sklearn.metrics import mean_squared_error as MSE

    # Read the CSV file
    aero = pd.read_csv('Sensordata.csv')

    # Finding the shape of the dataframe
    print(aero.shape)
    print(aero.head())

    # aero = aero.drop_duplicates()

    # Finding the shape of the dataframe
    # print(aero.shape)
    # Having a look at the data
    print(aero.head())

    # Brief data preprocessing
    aero.drop_duplicates()
    aero["Date"] = pd.to_datetime(aero["Date"], format=r"%d/%m/%Y %H:%M:%S", errors='coerce', dayfirst=True)
    # Set kolom 'Date' sebagai indeks DataFrame
    aero = aero.drop_duplicates("Date").set_index("Date").resample("h").mean()

    # Ubah format indeks menjadi string dengan format yang diinginkan
    aero.index = aero.index.strftime('%d/%m/%Y %H:%M:%S')
    aero = aero.dropna()
    print(len(aero))
    from sklearn.model_selection import train_test_split

    # Specify the test size (in this case, 20% for testing, 80% for training)
    test_size = 0.2

    # Use train_test_split to perform the split
    train_aero, test_aero = train_test_split(aero["TDS"], test_size=test_size, random_state=1)

    # models HWES ADD and MUL TDS
    model_add= HWES(train_aero, seasonal_periods=24, trend='mul', seasonal='mul')
    fitted_add = model_add.fit()
    print(fitted_add.summary())
    pickle.dump(fitted_add,open('hwsTDSforecast.pkl','wb'))

    # Forecast the next 24 time points
    forecast = fitted_add.forecast(len(aero))

    print(forecast)

    # Create a DataFrame for the forecast
    forecast_data = pd.DataFrame({'Date': aero.index,'TDS_Prediction': forecast})

    # Convert 'Date_Time' column to Date_Timetime format with the correct format specified
    forecast_data['Date'] = pd.to_datetime(forecast_data['Date'].str.strip(), format='%d/%m/%Y %H:%M:%S')
    aero.index = pd.to_datetime(aero.index, format='%d/%m/%Y %H:%M:%S')
    forecast_data= forecast_data.set_index('Date')
    # Resample the data to hourly intervals and take the mean
    #aero = aero.resample('H').mean()
    # Group by hour and calculate the mean for each hour
    aero = aero.groupby(pd.Grouper(freq='H')).mean()
    aero.index = aero.index.strftime('%d/%m/%Y %H:%M:%S')
    aero.index = aero.index.astype(str)
    aero=aero.dropna()
    aero

    forecast_data = forecast_data.groupby(pd.Grouper(freq='H')).mean()
    forecast_data.index = forecast_data.index.strftime('%d/%m/%Y %H:%M:%S')
    forecast_data.index = forecast_data.index.astype(str)
    forecast_data=forecast_data.dropna()
    forecast_data

    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime

    # Define the start and end Date_Times for the desired Date_Time range
    start_date = timedate['start_date'].iloc[0]
    end_date = timedate['end_date'].iloc[0]

    # Get the input from the user
    #start_Date_Time_str = input("Masukkan tanggal mulai (format DD/MM/YYYY HH:MM:SS): ")
    #end_Date_Time_str = input("Masukkan tanggal akhir (format DD/MM/YYYY HH:MM:SS): ")
    start_Date_Time_str = datastart
    end_Date_Time_str = dataend

    # Convert the input strings to datetime objects
    start_Date_Time = start_Date_Time_str
    end_Date_Time = end_Date_Time_str

    # Slice the DataFrame to include only data within the specified Date_Time range
    filtered_data = forecast_data[(forecast_data.index >= start_Date_Time) & (forecast_data.index <= end_Date_Time)]
    aero_filtered = aero[(aero.index >= start_Date_Time) & (aero.index <= end_Date_Time)]

    # Convert the index to a timezone-aware Datetime index
    filtered_data.index = pd.to_datetime(filtered_data.index)
    aero_filtered.index = pd.to_datetime(aero_filtered.index)

    # Group by hourly frequency and calculate the mean
    filtered_data = filtered_data.groupby(pd.Grouper(freq='H')).mean().dropna()
    aero_filtered = aero_filtered.groupby(pd.Grouper(freq='H')).mean().dropna()

    from datetime import datetime
    import pandas as pd

    # Dapatkan waktu dan tanggal saat ini dari sistem
    current_datetime = datetime.now()

    # Konversi waktu dan tanggal saat ini ke format yang diinginkan
    #input_date_time = current_datetime.strftime('%d/%m/%Y %H:%M:%S')

    # Konversi input_date_time ke format datetime
    #desired_date_time = pd.to_datetime(input_date_time)
    desired_date_time = pd.to_datetime(current_datetime)


    # Forecast menggunakan model Holt-Winters untuk waktu tertentu
    desired_forecast = fitted_add.forecast(steps=1).iloc[0]  # Prediksi 1 langkah ke depan (waktu yang diinginkan)
    print(desired_forecast)
    # Membuat DataFrame untuk hasil prediksi
    desired_forecast_data = pd.DataFrame({
        'Date': [desired_date_time],
        'TDS_Prediction': int(round(desired_forecast))
    })

    # Konversi 'Date' ke format datetime dengan format yang benar
    desired_forecast_data['Date'] = pd.to_datetime(desired_forecast_data['Date'])
    desired_forecast_data.index = pd.to_datetime(desired_forecast_data['Date'])
    print(desired_forecast_data)

    # Plot the original data and the forecast
    plt.figure(figsize=(20, 8))

    # Plot original data
    plt.plot(aero_filtered.index, aero_filtered['TDS'], label='Original Data')

    # Plot TDS Prediction (assuming you have 'TDS_Prediction' column in filtered_data)
    plt.plot(filtered_data.index, filtered_data['TDS_Prediction'], label='TDS Prediction', linestyle='-')

    # Customize the plot appearance
    plt.title('Original Data and TDS Prediction')
    plt.xlabel('Waktu')
    plt.ylabel('TDS (PPM_Parts Per Million)')
    plt.legend()

    # Set y-axis to start from 0 and end at 7
    plt.ylim([0, 2000])

    # Customize x-axis to start from 0 and display dates and times evenly
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # Format for displaying time
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Set interval for displaying hours

    # Show the plot
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    # Highlight the TDS range of 800 to 1200
    plt.axhspan(800, 1200, color='green', alpha=0.3)

    # Add vertical grid lines for each hour between 06:00 and 15:00
    current_time = pd.to_datetime(start_Date_Time)
    end_time = pd.to_datetime(end_Date_Time)

    while current_time <= end_time:
        day_start = current_time.replace(hour=6, minute=0, second=0)
        day_end = current_time.replace(hour=15, minute=0, second=0)
        if day_end <= end_time:
            grid_time = day_start
            while grid_time <= day_end:
                plt.axvline(grid_time, color='gray', linestyle='--', alpha=0.5)
                grid_time += pd.Timedelta(hours=1)
        current_time += pd.Timedelta(days=1)

    # Set the x-axis tick label colors for the range 06:00 to 15:00
    current_time = pd.to_datetime(start_Date_Time)
    while current_time <= end_time:
        day_start = current_time.replace(hour=6, minute=0, second=0)
        day_end = current_time.replace(hour=15, minute=0, second=0)
        if day_end <= end_time:
            # Find the x-tick positions within the range and change their color to red
            ticks = ax.get_xticks()
            labels = ax.get_xticklabels()
            for i, tick in enumerate(ticks):
                tick_time = mdates.num2date(tick)
                tick_time = tick_time.replace(tzinfo=None)  # Ensure tick_time is timezone-naive
                if day_start <= tick_time <= day_end:
                    labels[i].set_color('red')
        current_time += pd.Timedelta(days=1)

    # Update tick labels
    ax.set_xticklabels(labels)
    plt.tight_layout()  # Adjust layout to prevent cropping of labels
    # Save the plot as an image
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)

    # Save the image to a local folder
    folder_path = "A:/platihanPy/aeros/aeroforcast/static"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, "plotTDS.png")
    with open(file_path, "wb") as file:
        file.write(bytes_image.getvalue())

    plt.show()

    import numpy as np

    TDS_actual = aero_filtered['TDS']
    TDS_predicted = filtered_data['TDS_Prediction']
    MSE_value = np.mean((TDS_actual - TDS_predicted)**2)
    RMSE_valueAll = np.sqrt(MSE_value)

    print(f'Root Mean Squared Error = {RMSE_valueAll}')
    return(bytes_image)
tds_sensor()

def pH_sensor():
    import pandas as pd
    import numpy as np
    import io
    import os
    import json
    import pickle
    from matplotlib import pyplot as plt
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
    from sklearn.metrics import mean_squared_error as MSE

    # Read the CSV file
    aero = pd.read_csv('Sensordata.csv')

    # Finding the shape of the dataframe
    print(aero.shape)
    print(aero.head())

    # aero = aero.drop_duplicates()

    # Finding the shape of the dataframe
    # print(aero.shape)
    # Having a look at the data
    print(aero.head())

    # Brief data preprocessing
    aero.drop_duplicates()
    aero["Date"] = pd.to_datetime(aero["Date"], format=r"%d/%m/%Y %H:%M:%S", errors='coerce', dayfirst=True)
    # Set kolom 'Date' sebagai indeks DataFrame
    aero = aero.drop_duplicates("Date").set_index("Date").resample("h").mean()

    # Ubah format indeks menjadi string dengan format yang diinginkan
    aero.index = aero.index.strftime('%d/%m/%Y %H:%M:%S')
    aero = aero.dropna()
    print(len(aero))
    from sklearn.model_selection import train_test_split

    # Specify the test size (in this case, 20% for testing, 80% for training)
    test_size = 0.2

    # Use train_test_split to perform the split
    train_aero, test_aero = train_test_split(aero["pH"], test_size=test_size, random_state=1)

    # models HWES ADD and MUL pH
    model_add= HWES(train_aero, seasonal_periods=24, trend='add', seasonal='add')
    fitted_add = model_add.fit()
    print(fitted_add.summary())
    pickle.dump(fitted_add,open('hwspHforecast.pkl','wb'))

    # Forecast the next 24 time points
    forecast = fitted_add.forecast(len(aero))

    print(forecast)

    # Create a DataFrame for the forecast
    forecast_data = pd.DataFrame({'Date': aero.index,'pH_Prediction': forecast})

    # Convert 'Date_Time' column to Date_Timetime format with the correct format specified
    forecast_data['Date'] = pd.to_datetime(forecast_data['Date'].str.strip(), format='%d/%m/%Y %H:%M:%S')
    aero.index = pd.to_datetime(aero.index, format='%d/%m/%Y %H:%M:%S')
    forecast_data= forecast_data.set_index('Date')
    # Resample the data to hourly intervals and take the mean
    #aero = aero.resample('H').mean()
    # Group by hour and calculate the mean for each hour
    aero = aero.groupby(pd.Grouper(freq='H')).mean()
    aero.index = aero.index.strftime('%d/%m/%Y %H:%M:%S')
    aero.index = aero.index.astype(str)
    aero=aero.dropna()
    aero

    forecast_data = forecast_data.groupby(pd.Grouper(freq='H')).mean()
    forecast_data.index = forecast_data.index.strftime('%d/%m/%Y %H:%M:%S')
    forecast_data.index = forecast_data.index.astype(str)
    forecast_data=forecast_data.dropna()
    forecast_data

    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime

    # Define the start and end Date_Times for the desired Date_Time range
    start_date = timedate['start_date'].iloc[0]
    end_date = timedate['end_date'].iloc[0]

    # Get the input from the user
    #start_Date_Time_str = input("Masukkan tanggal mulai (format DD/MM/YYYY HH:MM:SS): ")
    #end_Date_Time_str = input("Masukkan tanggal akhir (format DD/MM/YYYY HH:MM:SS): ")
    start_Date_Time_str = datastart
    end_Date_Time_str = dataend

    # Convert the input strings to datetime objects
    start_Date_Time = start_Date_Time_str
    end_Date_Time = end_Date_Time_str

    # Slice the DataFrame to include only data within the specified Date_Time range
    filtered_data = forecast_data[(forecast_data.index >= start_Date_Time) & (forecast_data.index <= end_Date_Time)]
    aero_filtered = aero[(aero.index >= start_Date_Time) & (aero.index <= end_Date_Time)]

    # Convert the index to a timezone-aware Datetime index
    filtered_data.index = pd.to_datetime(filtered_data.index)
    aero_filtered.index = pd.to_datetime(aero_filtered.index)

    # Group by hourly frequency and calculate the mean
    filtered_data = filtered_data.groupby(pd.Grouper(freq='H')).mean().dropna()
    aero_filtered = aero_filtered.groupby(pd.Grouper(freq='H')).mean().dropna()

    from datetime import datetime
    import pandas as pd

    # Dapatkan waktu dan tanggal saat ini dari sistem
    current_datetime = datetime.now()

    # Konversi waktu dan tanggal saat ini ke format yang diinginkan
    #input_date_time = current_datetime.strftime('%d/%m/%Y %H:%M:%S')

    # Konversi input_date_time ke format datetime
    #desired_date_time = pd.to_datetime(input_date_time)
    desired_date_time = pd.to_datetime(current_datetime)


    # Forecast menggunakan model Holt-Winters untuk waktu tertentu
    desired_forecast = fitted_add.forecast(steps=1).iloc[0]  # Prediksi 1 langkah ke depan (waktu yang diinginkan)
    print(desired_forecast)
    # Membuat DataFrame untuk hasil prediksi
    desired_forecast_data = pd.DataFrame({
        'Date': [desired_date_time],
        'pH_Prediction': int(round(desired_forecast))
    })

    # Konversi 'Date' ke format datetime dengan format yang benar
    desired_forecast_data['Date'] = pd.to_datetime(desired_forecast_data['Date'])
    desired_forecast_data.index = pd.to_datetime(desired_forecast_data['Date'])
    print(desired_forecast_data)

    # Plot the original data and the forecast
    plt.figure(figsize=(20, 8))

    # Plot original data
    plt.plot(aero_filtered.index, aero_filtered['pH'], label='Original Data')

    # Plot pH Prediction (assuming you have 'pH_Prediction' column in filtered_data)
    plt.plot(filtered_data.index, filtered_data['pH_Prediction'], label='pH Prediction', linestyle='-')

    # Customize the plot appearance
    plt.title('Original Data and pH Prediction')
    plt.xlabel('Waktu')
    plt.ylabel('pH')
    plt.legend()

    # Set y-axis to start from 0 and end at 7
    plt.ylim([0, 10])

    # Customize x-axis to start from 0 and display dates and times evenly
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # Format for displaying time
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Set interval for displaying hours

    # Show the plot
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.axhspan(5.5, 6.5, color='green', alpha=0.3)

    # Add vertical grid lines for each hour between 06:00 and 15:00
    current_time = pd.to_datetime(start_Date_Time)
    end_time = pd.to_datetime(end_Date_Time)

    while current_time <= end_time:
        day_start = current_time.replace(hour=6, minute=0, second=0)
        day_end = current_time.replace(hour=15, minute=0, second=0)
        if day_end <= end_time:
            grid_time = day_start
            while grid_time <= day_end:
                plt.axvline(grid_time, color='gray', linestyle='--', alpha=0.5)
                grid_time += pd.Timedelta(hours=1)
        current_time += pd.Timedelta(days=1)

    # Set the x-axis tick label colors for the range 06:00 to 15:00
    current_time = pd.to_datetime(start_Date_Time)
    while current_time <= end_time:
        day_start = current_time.replace(hour=6, minute=0, second=0)
        day_end = current_time.replace(hour=15, minute=0, second=0)
        if day_end <= end_time:
            # Find the x-tick positions within the range and change their color to red
            ticks = ax.get_xticks()
            labels = ax.get_xticklabels()
            for i, tick in enumerate(ticks):
                tick_time = mdates.num2date(tick)
                tick_time = tick_time.replace(tzinfo=None)  # Ensure tick_time is timezone-naive
                if day_start <= tick_time <= day_end:
                    labels[i].set_color('red')
        current_time += pd.Timedelta(days=1)

    # Update tick labels
    ax.set_xticklabels(labels)
    plt.tight_layout()  # Adjust layout to prevent cropping of labels
    # Save the plot as an image
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)

    # Save the image to a local folder
    folder_path = "A:/platihanPy/aeros/aeroforcast/static"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, "plotpH.png")
    with open(file_path, "wb") as file:
        file.write(bytes_image.getvalue())

    plt.show()

    import numpy as np

    pH_actual = aero_filtered['pH']
    pH_predicted = filtered_data['pH_Prediction']
    MSE_value = np.mean((pH_actual - pH_predicted)**2)
    RMSE_valueAll = np.sqrt(MSE_value)

    print(f'Root Mean Squared Error = {RMSE_valueAll}')
    return(bytes_image)
pH_sensor()

def Temp_sensor():
    import pandas as pd
    import numpy as np
    import io
    import os
    import json
    import pickle
    from matplotlib import pyplot as plt
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
    from sklearn.metrics import mean_squared_error as MSE

    # Read the CSV file
    aero = pd.read_csv('Sensordata.csv')

    # Finding the shape of the dataframe
    print(aero.shape)
    print(aero.head())

    # aero = aero.drop_duplicates()

    # Finding the shape of the dataframe
    # print(aero.shape)
    # Having a look at the data
    print(aero.head())

    # Brief data preprocessing
    aero.drop_duplicates()
    aero["Date"] = pd.to_datetime(aero["Date"], format=r"%d/%m/%Y %H:%M:%S", errors='coerce', dayfirst=True)
    # Set kolom 'Date' sebagai indeks DataFrame
    aero = aero.drop_duplicates("Date").set_index("Date").resample("h").mean()

    # Ubah format indeks menjadi string dengan format yang diinginkan
    aero.index = aero.index.strftime('%d/%m/%Y %H:%M:%S')
    aero = aero.dropna()
    print(len(aero))
    from sklearn.model_selection import train_test_split

    # Specify the test size (in this case, 20% for testing, 80% for training)
    test_size = 0.2

    # Use train_test_split to perform the split
    train_aero, test_aero = train_test_split(aero["Temp"], test_size=test_size, random_state=1)

    # models HWES ADD and MUL Temp
    model_add= HWES(train_aero, seasonal_periods=24, trend='add', seasonal='add')
    fitted_add = model_add.fit()
    print(fitted_add.summary())
    pickle.dump(fitted_add,open('hwsTempforecast.pkl','wb'))

    # Forecast the next 24 time points
    forecast = fitted_add.forecast(len(aero))

    print(forecast)

    # Create a DataFrame for the forecast
    forecast_data = pd.DataFrame({'Date': aero.index,'Temp_Prediction': forecast})

    # Convert 'Date_Time' column to Date_Timetime format with the correct format specified
    forecast_data['Date'] = pd.to_datetime(forecast_data['Date'].str.strip(), format='%d/%m/%Y %H:%M:%S')
    aero.index = pd.to_datetime(aero.index, format='%d/%m/%Y %H:%M:%S')
    forecast_data= forecast_data.set_index('Date')
    # Resample the data to hourly intervals and take the mean
    #aero = aero.resample('H').mean()
    # Group by hour and calculate the mean for each hour
    aero = aero.groupby(pd.Grouper(freq='H')).mean()
    aero.index = aero.index.strftime('%d/%m/%Y %H:%M:%S')
    aero.index = aero.index.astype(str)
    aero=aero.dropna()
    aero

    forecast_data = forecast_data.groupby(pd.Grouper(freq='H')).mean()
    forecast_data.index = forecast_data.index.strftime('%d/%m/%Y %H:%M:%S')
    forecast_data.index = forecast_data.index.astype(str)
    forecast_data=forecast_data.dropna()
    forecast_data

    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime

    # Define the start and end Date_Times for the desired Date_Time range
    start_date = timedate['start_date'].iloc[0]
    end_date = timedate['end_date'].iloc[0]

    # Get the input from the user
    #start_Date_Time_str = input("Masukkan tanggal mulai (format DD/MM/YYYY HH:MM:SS): ")
    #end_Date_Time_str = input("Masukkan tanggal akhir (format DD/MM/YYYY HH:MM:SS): ")
    start_Date_Time_str = datastart
    end_Date_Time_str = dataend

    # Convert the input strings to datetime objects
    start_Date_Time = start_Date_Time_str
    end_Date_Time = end_Date_Time_str

    # Slice the DataFrame to include only data within the specified Date_Time range
    filtered_data = forecast_data[(forecast_data.index >= start_Date_Time) & (forecast_data.index <= end_Date_Time)]
    aero_filtered = aero[(aero.index >= start_Date_Time) & (aero.index <= end_Date_Time)]

    # Convert the index to a timezone-aware Datetime index
    filtered_data.index = pd.to_datetime(filtered_data.index)
    aero_filtered.index = pd.to_datetime(aero_filtered.index)

    # Group by hourly frequency and calculate the mean
    filtered_data = filtered_data.groupby(pd.Grouper(freq='H')).mean().dropna()
    aero_filtered = aero_filtered.groupby(pd.Grouper(freq='H')).mean().dropna()

    from datetime import datetime
    import pandas as pd

    # Dapatkan waktu dan tanggal saat ini dari sistem
    current_datetime = datetime.now()

    # Konversi waktu dan tanggal saat ini ke format yang diinginkan
    #input_date_time = current_datetime.strftime('%d/%m/%Y %H:%M:%S')

    # Konversi input_date_time ke format datetime
    #desired_date_time = pd.to_datetime(input_date_time)
    desired_date_time = pd.to_datetime(current_datetime)


    # Forecast menggunakan model Holt-Winters untuk waktu tertentu
    desired_forecast = fitted_add.forecast(steps=1).iloc[0]  # Prediksi 1 langkah ke depan (waktu yang diinginkan)
    print(desired_forecast)
    # Membuat DataFrame untuk hasil prediksi
    desired_forecast_data = pd.DataFrame({
        'Date': [desired_date_time],
        'Temp_Prediction': int(round(desired_forecast))
    })

    # Konversi 'Date' ke format datetime dengan format yang benar
    desired_forecast_data['Date'] = pd.to_datetime(desired_forecast_data['Date'])
    desired_forecast_data.index = pd.to_datetime(desired_forecast_data['Date'])
    print(desired_forecast_data)

    # Plot the original data and the forecast
    plt.figure(figsize=(20, 8))

    # Plot original data
    plt.plot(aero_filtered.index, aero_filtered['Temp'], label='Original Data')

    # Plot Temp Prediction (assuming you have 'Temp_Prediction' column in filtered_data)
    plt.plot(filtered_data.index, filtered_data['Temp_Prediction'], label='Temp Prediction', linestyle='-')

    # Customize the plot appearance
    plt.title('Original Data and Temp Prediction')
    plt.xlabel('Waktu')
    plt.ylabel('Temp (Temperature)')
    plt.legend()

    # Set y-axis to start from 0 and end at 7
    plt.ylim([0, 50])

    # Customize x-axis to start from 0 and display dates and times evenly
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # Format for displaying time
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Set interval for displaying hours

    # Show the plot
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    # Highlight the TDS range of 800 to 1200
    plt.axhspan(18, 29, color='green', alpha=0.3)

    # Add vertical grid lines for each hour between 06:00 and 15:00
    current_time = pd.to_datetime(start_Date_Time)
    end_time = pd.to_datetime(end_Date_Time)

    while current_time <= end_time:
        day_start = current_time.replace(hour=6, minute=0, second=0)
        day_end = current_time.replace(hour=15, minute=0, second=0)
        if day_end <= end_time:
            grid_time = day_start
            while grid_time <= day_end:
                plt.axvline(grid_time, color='gray', linestyle='--', alpha=0.5)
                grid_time += pd.Timedelta(hours=1)
        current_time += pd.Timedelta(days=1)

    # Set the x-axis tick label colors for the range 06:00 to 15:00
    current_time = pd.to_datetime(start_Date_Time)
    while current_time <= end_time:
        day_start = current_time.replace(hour=6, minute=0, second=0)
        day_end = current_time.replace(hour=15, minute=0, second=0)
        if day_end <= end_time:
            # Find the x-tick positions within the range and change their color to red
            ticks = ax.get_xticks()
            labels = ax.get_xticklabels()
            for i, tick in enumerate(ticks):
                tick_time = mdates.num2date(tick)
                tick_time = tick_time.replace(tzinfo=None)  # Ensure tick_time is timezone-naive
                if day_start <= tick_time <= day_end:
                    labels[i].set_color('red')
        current_time += pd.Timedelta(days=1)

    # Update tick labels
    ax.set_xticklabels(labels)
    plt.tight_layout()  # Adjust layout to prevent cropping of labels
    # Save the plot as an image
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)

    # Save the image to a local folder
    folder_path = "A:/platihanPy/aeros/aeroforcast/static"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, "plotTemp.png")
    with open(file_path, "wb") as file:
        file.write(bytes_image.getvalue())

    plt.show()

    import numpy as np

    Temp_actual = aero_filtered['Temp']
    Temp_predicted = filtered_data['Temp_Prediction']
    MSE_value = np.mean((Temp_actual - Temp_predicted)**2)
    RMSE_valueAll = np.sqrt(MSE_value)

    print(f'Root Mean Squared Error = {RMSE_valueAll}')
    return(bytes_image)
Temp_sensor()

def Hum_sensor():
    import pandas as pd
    import numpy as np
    import io
    import os
    import json
    import pickle
    from matplotlib import pyplot as plt
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
    from sklearn.metrics import mean_squared_error as MSE

    # Read the CSV file
    aero = pd.read_csv('Sensordata.csv')

    # Finding the shape of the dataframe
    print(aero.shape)
    print(aero.head())

    # aero = aero.drop_duplicates()

    # Finding the shape of the dataframe
    # print(aero.shape)
    # Having a look at the data
    print(aero.head())

    # Brief data preprocessing
    aero.drop_duplicates()
    aero["Date"] = pd.to_datetime(aero["Date"], format=r"%d/%m/%Y %H:%M:%S", errors='coerce', dayfirst=True)
    # Set kolom 'Date' sebagai indeks DataFrame
    aero = aero.drop_duplicates("Date").set_index("Date").resample("h").mean()

    # Ubah format indeks menjadi string dengan format yang diinginkan
    aero.index = aero.index.strftime('%d/%m/%Y %H:%M:%S')
    aero = aero.dropna()
    print(len(aero))
    from sklearn.model_selection import train_test_split

    # Specify the test size (in this case, 20% for testing, 80% for training)
    test_size = 0.2

    # Use train_test_split to perform the split
    train_aero, test_aero = train_test_split(aero["Hum"], test_size=test_size, random_state=1)

    # models HWES ADD and MUL Hum
    model_add= HWES(train_aero, seasonal_periods=24, trend='add', seasonal='add')
    fitted_add = model_add.fit()
    print(fitted_add.summary())
    pickle.dump(fitted_add,open('hwsHumforecast.pkl','wb'))

    # Forecast the next 24 time points
    forecast = fitted_add.forecast(len(aero))

    print(forecast)

    # Create a DataFrame for the forecast
    forecast_data = pd.DataFrame({'Date': aero.index,'Hum_Prediction': forecast})

    # Convert 'Date_Time' column to Date_Timetime format with the correct format specified
    forecast_data['Date'] = pd.to_datetime(forecast_data['Date'].str.strip(), format='%d/%m/%Y %H:%M:%S')
    aero.index = pd.to_datetime(aero.index, format='%d/%m/%Y %H:%M:%S')
    forecast_data= forecast_data.set_index('Date')
    # Resample the data to hourly intervals and take the mean
    #aero = aero.resample('H').mean()
    # Group by hour and calculate the mean for each hour
    aero = aero.groupby(pd.Grouper(freq='H')).mean()
    aero.index = aero.index.strftime('%d/%m/%Y %H:%M:%S')
    aero.index = aero.index.astype(str)
    aero=aero.dropna()
    aero

    forecast_data = forecast_data.groupby(pd.Grouper(freq='H')).mean()
    forecast_data.index = forecast_data.index.strftime('%d/%m/%Y %H:%M:%S')
    forecast_data.index = forecast_data.index.astype(str)
    forecast_data=forecast_data.dropna()
    forecast_data

    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime

    # Define the start and end Date_Times for the desired Date_Time range
    start_date = timedate['start_date'].iloc[0]
    end_date = timedate['end_date'].iloc[0]

    # Get the input from the user
    #start_Date_Time_str = input("Masukkan tanggal mulai (format DD/MM/YYYY HH:MM:SS): ")
    #end_Date_Time_str = input("Masukkan tanggal akhir (format DD/MM/YYYY HH:MM:SS): ")
    start_Date_Time_str = datastart
    end_Date_Time_str = dataend

    # Convert the input strings to datetime objects
    start_Date_Time = start_Date_Time_str
    end_Date_Time = end_Date_Time_str

    # Slice the DataFrame to include only data within the specified Date_Time range
    filtered_data = forecast_data[(forecast_data.index >= start_Date_Time) & (forecast_data.index <= end_Date_Time)]
    aero_filtered = aero[(aero.index >= start_Date_Time) & (aero.index <= end_Date_Time)]

    # Convert the index to a timezone-aware Datetime index
    filtered_data.index = pd.to_datetime(filtered_data.index)
    aero_filtered.index = pd.to_datetime(aero_filtered.index)

    # Group by hourly frequency and calculate the mean
    filtered_data = filtered_data.groupby(pd.Grouper(freq='H')).mean().dropna()
    aero_filtered = aero_filtered.groupby(pd.Grouper(freq='H')).mean().dropna()

    from datetime import datetime
    import pandas as pd

    # Dapatkan waktu dan tanggal saat ini dari sistem
    current_datetime = datetime.now()

    # Konversi waktu dan tanggal saat ini ke format yang diinginkan
    #input_date_time = current_datetime.strftime('%d/%m/%Y %H:%M:%S')

    # Konversi input_date_time ke format datetime
    #desired_date_time = pd.to_datetime(input_date_time)
    desired_date_time = pd.to_datetime(current_datetime)


    # Forecast menggunakan model Holt-Winters untuk waktu tertentu
    desired_forecast = fitted_add.forecast(steps=1).iloc[0]  # Prediksi 1 langkah ke depan (waktu yang diinginkan)
    print(desired_forecast)
    # Membuat DataFrame untuk hasil prediksi
    desired_forecast_data = pd.DataFrame({
        'Date': [desired_date_time],
        'Hum_Prediction': int(round(desired_forecast))
    })

    # Konversi 'Date' ke format datetime dengan format yang benar
    desired_forecast_data['Date'] = pd.to_datetime(desired_forecast_data['Date'])
    desired_forecast_data.index = pd.to_datetime(desired_forecast_data['Date'])
    print(desired_forecast_data)

    # Plot the original data and the forecast
    plt.figure(figsize=(20, 8))

    # Plot original data
    plt.plot(aero_filtered.index, aero_filtered['Hum'], label='Original Data')

    # Plot Hum Prediction (assuming you have 'Hum_Prediction' column in filtered_data)
    plt.plot(filtered_data.index, filtered_data['Hum_Prediction'], label='Hum Prediction', linestyle='-')

    # Customize the plot appearance
    plt.title('Original Data and Hum Prediction')
    plt.xlabel('Waktu')
    plt.ylabel('Hum (Humadity)')
    plt.legend()

    # Set y-axis to start from 0 and end at 7
    plt.ylim([0, 120])

    # Customize x-axis to start from 0 and display dates and times evenly
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # Format for displaying time
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Set interval for displaying hours

    # Show the plot
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    # Highlight the TDS range of 800 to 1200
    plt.axhspan(60, 80, color='green', alpha=0.3)

    # Add vertical grid lines for each hour between 06:00 and 15:00
    current_time = pd.to_datetime(start_Date_Time)
    end_time = pd.to_datetime(end_Date_Time)

    while current_time <= end_time:
        day_start = current_time.replace(hour=6, minute=0, second=0)
        day_end = current_time.replace(hour=15, minute=0, second=0)
        if day_end <= end_time:
            grid_time = day_start
            while grid_time <= day_end:
                plt.axvline(grid_time, color='gray', linestyle='--', alpha=0.5)
                grid_time += pd.Timedelta(hours=1)
        current_time += pd.Timedelta(days=1)

    # Set the x-axis tick label colors for the range 06:00 to 15:00
    current_time = pd.to_datetime(start_Date_Time)
    while current_time <= end_time:
        day_start = current_time.replace(hour=6, minute=0, second=0)
        day_end = current_time.replace(hour=15, minute=0, second=0)
        if day_end <= end_time:
            # Find the x-tick positions within the range and change their color to red
            ticks = ax.get_xticks()
            labels = ax.get_xticklabels()
            for i, tick in enumerate(ticks):
                tick_time = mdates.num2date(tick)
                tick_time = tick_time.replace(tzinfo=None)  # Ensure tick_time is timezone-naive
                if day_start <= tick_time <= day_end:
                    labels[i].set_color('red')
        current_time += pd.Timedelta(days=1)

    # Update tick labels
    ax.set_xticklabels(labels)
    plt.tight_layout()  # Adjust layout to prevent cropping of labels
    # Save the plot as an image
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)

    # Save the image to a local folder
    folder_path = "A:/platihanPy/aeros/aeroforcast/static"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, "plotHum.png")
    with open(file_path, "wb") as file:
        file.write(bytes_image.getvalue())

    plt.show()

    import numpy as np

    Hum_actual = aero_filtered['Hum']
    Hum_predicted = filtered_data['Hum_Prediction']
    MSE_value = np.mean((Hum_actual - Hum_predicted)**2)
    RMSE_valueAll = np.sqrt(MSE_value)

    print(f'Root Mean Squared Error = {RMSE_valueAll}')
    return(bytes_image)
Hum_sensor()