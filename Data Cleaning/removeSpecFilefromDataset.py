#remove the given files from the folders and subsequently from the jsons from the dataset

import os
import json

def remove_files_from_folders(subfolders, files_to_remove):
    for subfolder in subfolders:
        images_folder = os.path.join(subfolder, 'images')
        labels_json_path = os.path.join(subfolder, 'labels.json')

        # Check if the subfolder has the necessary structure
        if os.path.exists(images_folder) and os.path.exists(labels_json_path):
            # Read labels.json
            with open(labels_json_path, 'r') as json_file:
                labels_data = json.load(json_file)

            # Remove specified files from the images folder and update labels.json
            for file_to_remove in files_to_remove:
                print(file_to_remove)
                print(type(file_to_remove))
                image_path = os.path.join(images_folder, file_to_remove)

                if os.path.exists(image_path):
                    # Remove image file
                    os.remove(image_path)

                # # Remove entry from labels.json if it exists
                # if file_to_remove in labels_data:
                #     del labels_data[file_to_remove]

            # Update labels.json
            with open(labels_json_path, 'w') as json_file:
                json.dump(labels_data, json_file, indent=2)

            print(f"Removed files from {subfolder}")

if __name__ == "__main__":
    # Replace 'path/to/your/subfolders' with the actual path to your subfolders
    # subfolders_path = 'path/to/your/subfolders'

    # List of subfolders
    subfolders = ['/data3/sreevatsa/Datasets/exp/dataset5_8124/val',
    '/data3/sreevatsa/Datasets/exp/dataset5_8124/test',
    '/data3/sreevatsa/Datasets/exp/dataset5_8124/train/Easy',
    '/data3/sreevatsa/Datasets/exp/dataset5_8124/train/Medium',
    '/data3/sreevatsa/Datasets/exp/dataset5_8124/train/Hard']
    # List of files to remove
    files_to_remove = ['306_1.jpg', '227_1.jpg', '306_56.jpg', '227_2.jpg', '227_60.jpg', '306_55.jpg', '306_2.jpg', '227_59.jpg', '304_2.jpg', '121_39.jpg', '228_63.jpg', '121_20.jpg', '121_16.jpg', '121_17.jpg', '121_19.jpg', '121_29.jpg', '121_25.jpg', '121_38.jpg', '121_22.jpg', '121_7.jpg', '320_27.jpg', '121_27.jpg', '121_5.jpg', '121_4.jpg', '121_8.jpg', '121_34.jpg', '121_13.jpg', '121_32.jpg', '121_40.jpg', '304_55.jpg', '121_18.jpg', '121_26.jpg', '121_35.jpg', '121_21.jpg', '121_37.jpg', '121_23.jpg', '305_1.jpg', '228_1.jpg', '121_14.jpg', '121_12.jpg', '121_24.jpg', '121_33.jpg', '228_64.jpg', '121_36.jpg', '228_2.jpg', '121_28.jpg']  # Add your file names here

    # Call the function
    remove_files_from_folders(subfolders, files_to_remove)

# import re
# import os
# def extract_paths_from_warnings(file_path):
#     with open(file_path, 'r') as file:
#         # Read all lines from the file
#         lines = file.readlines()

#     # Define a regular expression to match file paths in DecompressionBombWarning lines
#     warning_regex = re.compile(r'DecompressionBombWarning for file: (.+)')

#     # Extract file paths from the lines
#     paths = [warning_regex.search(line).group(1) for line in lines if warning_regex.search(line)]

#     paths = [os.path.basename(path) for path in paths]
#     return paths

# if __name__ == "__main__":
#     # Replace 'path/to/your/warnings.txt' with the actual path to your text file
#     warnings_file_path = '/data3/sreevatsa/Datasets/rr.txt'

#     # Call the function to extract paths
#     paths_list = extract_paths_from_warnings(warnings_file_path)

#     # Print the list of paths
#     print("List of paths:")
#     print(paths_list)
