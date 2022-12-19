import datetime
import multiprocessing as mp
import pickle
import sys
from tqdm import tqdm

from weather import get_weather, get_weather_analytical

lat = "43.84"
lng = "-79.33"


def get_irradiance():
    rows = []
    for year in sys.argv[2:]:
        data = get_weather_analytical(
            lat,
            lng,
            datetime.datetime.strptime(f"{year}-01-01", "%Y-%m-%d"),
            datetime.datetime.strptime(f"{year}-12-31", "%Y-%m-%d"),
        )
        if data is None:
            continue
        rows += list(
            zip(
                list(data["date"]),
                list(data["GlobalHorizontalIrradianceLocalDaytimeAvg"]),
            )
        )
    return rows


def get_weather_date(date):
    year_start = datetime.datetime.strptime(f"{date[:4]}0101", "%Y%m%d")
    date = datetime.datetime.strptime(date, "%Y%m%d")
    days = (year_start - date).days / 365
    weather = get_weather(lat, lng, date)
    if len(weather["validTimeUtc"]) != 24:
        return None

    features = [
        "relativeHumidity",
        "temperature",
        "windSpeed",
        "pressureMeanSeaLevel",
        "temperatureDewPoint",
        "uvIndex",
        "windGust",
        "windDirection",
        "temperatureFeelsLike",
    ]

    data = [[weather[feature][x] for feature in features] + [days] for x in range(24)]
    data = [[x if x is not None else -1 for x in y] for y in data]

    return data


def process(x):
    date, rad = x
    return get_weather_date(str(date)), rad


if __name__ == "__main__":
    irradiance = get_irradiance()

    pool = mp.Pool()
    data = [
        x
        for x in list(tqdm(pool.imap(process, irradiance), total=len(irradiance)))
        if x[0] is not None
    ]

    pickle.dump(data, open(sys.argv[1], "wb"))
