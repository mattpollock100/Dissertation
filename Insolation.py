#%%
import ephem
import pandas as pd
from datetime import datetime, timedelta

def calculate_insolation(latitude, longitude, year):
    if year < 0:
        year_str = f"{abs(year)} BC"
        year = abs(year) + 1
    else:
        year_str = f"{year} AD"
    
    # Generate dates manually for the specified year
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    current_date = start_date

    daily_insolation = []

    while current_date <= end_date:
        observer = ephem.Observer()
        observer.lat = str(latitude)
        observer.lon = str(longitude)
        observer.date = ephem.Date(current_date)

        # Calculate solar position and irradiance
        sun = ephem.Sun(observer)
        sunrise = observer.previous_rising(sun)
        sunset = observer.next_setting(sun)

        total_insolation = 0
        time_step = timedelta(minutes=60)

        current_time = ephem.Date(sunrise)
        while current_time < ephem.Date(sunset):
            observer.date = current_time
            sun.compute(observer)
            altitude = sun.alt

            if altitude > 0:
                # Simple model: use altitude to estimate DNI
                dni = 1361 * max(0, altitude / (90 * ephem.degree))  # Adjust this calculation for a more accurate model
                total_insolation += dni * (time_step.seconds / 3600)

            # Increment current_time by time_step
            current_time = ephem.Date(current_time.datetime() + time_step)

        daily_insolation.append((current_date, total_insolation))
        current_date += timedelta(days=1)
    
    # Create DataFrame
    insolation_df = pd.DataFrame(daily_insolation, columns=['Date', 'Insolation (W/m²)'])
    insolation_df['Year'] = year_str

    return insolation_df

# Example usage
latitude = 37.7749  # Latitude for San Francisco
longitude = -122.4194  # Longitude for San Francisco
year = -500  # 500 BC

insolation_df = calculate_insolation(latitude, longitude, year)
print(insolation_df)


# %%
#get average insolation for the year
insolation_df['Insolation (W/m²)'].mean()

# %%
