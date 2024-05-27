import pandas as pd

import os

directory_path = '/root/Speech_diarisation/NewData/NewData/Custom_Training_Set'

# Iterating through the directory
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    
    if (os.path.isdir(file_path) and filename[:9] == "corrected" ):
        print("Directory:"+ file_path )
        
# Data to be written to CSV
# data = {
#     "file": ["Alice", "Bob", "Charlie"],
#     "transcript": [30, 25, 35],  
# }

# # Creating a DataFrame
# df = pd.DataFrame(data)

# # Writing to CSV
# df.to_csv('training.csv', index=False)