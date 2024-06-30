



import os
import numpy as np
import pandas as pd

# Specify the directory you want to use
directory = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/ENSO Masks/'

# Loop over all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a .npy file
    if filename.endswith('.npy'):
        # Load the .npy file
        data = np.load(os.path.join(directory, filename))
        
        # Convert the numpy array to a pandas DataFrame
        df = pd.DataFrame(data)
        
        # Save the DataFrame as a .csv file
        df.to_csv(os.path.join(directory + 'CSV Versions', filename[:-4] + '.csv'), index=False)