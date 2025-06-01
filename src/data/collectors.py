from meteostat import Hourly, Point
from datetime import datetime
import pandas as pd
import os

location = Point(40.1828, 29.0668)
start = datetime(2019, 1, 1)
end = datetime(2023, 8, 27)

data = Hourly(location, start, end)
data_frame = data.fetch()
data_frame.reset_index(inplace=True)

raw_data_dir = os.path.join("data", "raw")
os.makedirs(raw_data_dir, exist_ok=True)

output_file = os.path.join(raw_data_dir, "bursa_hourly_weather_data.xlsx")
data_frame.to_excel(output_file, index=False)

print(data_frame.head(10))
print(f"Data saved to {output_file}")

weather_data = pd.read_excel("data/raw/bursa_hourly_weather_data.xlsx")
smfdb_data = pd.read_excel("data/raw/smfdb.xlsx")

weather_data = weather_data.drop(columns=["Time"])

merged_data = pd.concat([smfdb_data, weather_data], axis=1)

output_folder = "data/raw"
os.makedirs(output_folder, exist_ok=True)

output_path = os.path.join(output_folder, "merged_data.xlsx")
merged_data.to_excel(output_path, index=False)

print(f"Merging completed and saved to '{output_path}'")
