# import os
# import shutil
# import json

# def merge_images_and_jsons(source_folder, destination_folder, destination_json):
#     # Create the destination folder if it doesn't exist
#     if not os.path.exists(destination_folder):
#         os.makedirs(destination_folder)

#     # List all subdirectories in the source folder
#     subdirectories = [f.path for f in os.scandir(source_folder) if f.is_dir()]
#     print("Subdirectories are:",subdirectories)
#     # Iterate through each subdirectory
#     merged_json={}
#     for subdir in subdirectories:
#         # Check if the subdirectory has a 'ravi_gt' folder
#         ravi_gt_folder = os.path.join(subdir, 'ravi_style_gt')
#         if os.path.exists(ravi_gt_folder) and os.path.isdir(ravi_gt_folder):
#             # Process images
#             images_folder = os.path.join(ravi_gt_folder, 'images')
#             if os.path.exists(images_folder) and os.path.isdir(images_folder):
#                 image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
#                 for image_file in image_files:
#                     source_image_path = os.path.join(images_folder, image_file)
#                     destination_image_path = os.path.join(destination_folder, image_file)
#                     shutil.copy2(source_image_path, destination_image_path)

#             # Process JSON files
#             json_file_path = os.path.join(ravi_gt_folder, 'gt.json')
#             if os.path.exists(json_file_path):
#                 with open(json_file_path, 'r') as json_file:
#                     json_data = json.load(json_file)
#                     print("Json data is:",json_data)
#                     merged_json.update(json_data)
#                     # Assuming you have a list of JSON objects in each file
#                     # Append or merge these objects as needed
#                     # For example, if the 'gt.json' file contains a list, you can do:
#                     # merged_json_data.extend(json_data)
#                     # If it contains a dictionary, you might want to merge the dictionaries.
#         with open(destination_json,"w") as json_file:
#             json.dump(merged_json,json_file,indent=4)
#     print("Merge operation completed.")

# # Specify your source and destination folders
# source_folder = '/home/ndli19/docvisor/UseCaseDocsData/unzips'
# destination_folder = '/home/ndli19/docvisor/UseCaseDocsData/mergedData/mergedImages'
# destination_json='/home/ndli19/docvisor/UseCaseDocsData/mergedData/mergedJson.json'
# # Call the function
# merge_images_and_jsons(source_folder, destination_folder, destination_json)



import os
import shutil
import json
from json import JSONDecodeError

def merge_images_and_jsons(source_folder, destination_folder, destination_json, error_log_file):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # List all subdirectories in the source folder
    subdirectories = [f.path for f in os.scandir(source_folder) if f.is_dir()]
    print("Subdirectories are:",subdirectories)
    # Iterate through each subdirectory
    merged_json={}
    for subdir in subdirectories:
        # Check if the subdirectory has a 'ravi_gt' folder
        ravi_gt_folder = os.path.join(subdir, 'ravi_style_gt')
        if os.path.exists(ravi_gt_folder) and os.path.isdir(ravi_gt_folder):
            # Process images
            images_folder = os.path.join(ravi_gt_folder, 'images')
            if os.path.exists(images_folder) and os.path.isdir(images_folder):
                image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
                for image_file in image_files:
                    source_image_path = os.path.join(images_folder, image_file)
                    destination_image_path = os.path.join(destination_folder, image_file)
                    shutil.copy2(source_image_path, destination_image_path)

            # Process JSON files
            json_file_path = os.path.join(ravi_gt_folder, 'gt.json')
            print("Json file path is:",json_file_path)
            if os.path.exists(json_file_path):
                try:
                    with open(json_file_path, 'r') as json_file:
                        json_data = json.load(json_file)
                        # print("Json data is:",json_data)
                        merged_json.update(json_data)
                except JSONDecodeError:
                    with open(error_log_file, 'a') as error_log:
                        error_log.write(f"JSONDecodeError in folder: {subdir}\n")
                    continue

        with open(destination_json,"w") as json_file:
            json.dump(merged_json,json_file,indent=4)
    print("Merge operation completed.")

# Specify your source and destination folders
source_folder = '/home/ndli19/docvisor/UseCaseDocsData/unzips'
destination_folder = '/home/ndli19/docvisor/UseCaseDocsData/mergedData/mergedImages'
destination_json='/home/ndli19/docvisor/UseCaseDocsData/mergedData/mergedJson.json'
error_log_file = '/home/ndli19/docvisor/UseCaseDocsData/error_log_2.txt'
# Call the function
merge_images_and_jsons(source_folder, destination_folder, destination_json, error_log_file)