# update the file names in the json from .jpg/.png extension to .tiff extension

import json
import os

# Specify the path to your JSON file
json_file_path = '/data3/sreevatsa/Datasets/exp/dataset5_8124/train/Hard/labels.json'

# Read the JSON file
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# Update keys from '.jpg' to '.tiff'
updated_data = {os.path.splitext(key)[0] + '.tiff': value for key, value in data.items()}

# Specify the path to save the updated JSON file
updated_json_file_path = '/data3/sreevatsa/Datasets/exp/dataset5_8124/tiff_output/dataset5_8124/train/Hard/labels.json'

# Save the updated data to the JSON file
with open(updated_json_file_path, 'w') as updated_json_file:
    json.dump(updated_data, updated_json_file, indent=2)

print("File names updated and saved to", updated_json_file_path)
