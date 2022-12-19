import requests
import json
import datetime
import math
import io
import pandas as pd

API_KEY = "<APIKEY>"

def get_weather(lat, lng, date):
    start_date = date
    end_date = date + datetime.timedelta(days=1)

    url = "https://api.weather.com/v3/wx/hod/r1/direct?geocode={},{}&startDateTime={}&endDateTime={}&format=json&units=e&apiKey={}".format(
        lat, lng, start_date.strftime("%Y-%m-%dT%HZ"), end_date.strftime("%Y-%m-%dT%HZ"), API_KEY)

    resp = requests.get(url)
    resp.raise_for_status()

    weather = json.loads(resp.text)

    return weather

def get_weather_analytical(lat, lng, start_date, end_date):
    url = "https://api.weather.com/v3/wx/observations/historical/analytical/ext?geocode={},{}&startDate={}&endDate={}&format=csv&apiKey={}&productId=GlobalHorizontalIrradiance&language=en-US&units=s".format(
        lat, lng, start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"), API_KEY)

    resp = requests.get(url)
    resp.raise_for_status()

    try:
        weather = pd.read_csv(io.StringIO(resp.text))
        return weather
    except:
        return None

if __name__ == "__main__":
    print(get_weather_analytical("43.84", "-79.33", datetime.datetime.strptime('2020-01-01', '%Y-%m-%d'), datetime.datetime.strptime('2020-12-31', '%Y-%m-%d')))
    print(get_weather_analytical("43.84", "-79.33", datetime.datetime.strptime('2021-01-01', '%Y-%m-%d'), datetime.datetime.strptime('2021-12-31', '%Y-%m-%d')))
    print(json.dumps(get_weather("43.84", "-79.33", datetime.datetime.strptime('2021-01-01', '%Y-%m-%d'))))
