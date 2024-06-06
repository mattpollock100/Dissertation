# Common Functions

import scipy.ndimage as ndimage
import numpy as np
import cftime

def convert_dates(dataset, model_end_year):
    
    date_size = dataset.time.shape[0]
    start_year = (model_end_year - (date_size / 12))
    periods = date_size
    dates_xarray = [cftime.DatetimeProlepticGregorian(year=start_year + i // 12, month=i % 12 + 1, day=1) for i in range(periods)]
    dataset['time'] = dates_xarray

    return dataset, periods

# Define a function to find the number of ENSO events in a dataset
def find_enso_events(data, threshold, months_threshold):
    # Create a boolean array where True indicates that the anomaly is above 0.5
    above_threshold = data > threshold

    # Label contiguous True regions
    labeled_array, num_features = ndimage.label(above_threshold)

    # Count the size of each labeled region
    region_sizes = np.bincount(labeled_array.ravel())

    # Count the number of regions that have a size of 6 or more
    num_large_regions = np.count_nonzero(region_sizes >= months_threshold)

    avg_anomalies = []
    lengths = []

    for i in range(1, num_features + 1):
        # Get the size of the region
        size = np.count_nonzero(labeled_array == i)

        # If the size is greater than or equal to months_threshold
        if size >= months_threshold:
            # Append the size to lengths
            lengths.append(size)

            # Calculate the average anomaly in the region and append it to avg_anomalies
            avg_anomaly = data[labeled_array == i].mean()
            avg_anomalies.append(avg_anomaly)
    
    # Return a boolean of TRUE if in an ENSO, false if not
    mask_enso = np.zeros_like(above_threshold, dtype=bool)

    # Loop over each group of consecutive True values
    for i in range(1, num_features + 1):
        # If the group has 6 or more elements, set them to True in the mask
        if np.sum(labeled_array == i) >= months_threshold:
            mask_enso[labeled_array == i] = True

    return [num_large_regions, round(np.mean(avg_anomalies),3), round(np.mean(lengths),1)], mask_enso
 