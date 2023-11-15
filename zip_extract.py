import os
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir = '/home/user/docTR/extract_zips'
print(current_dir)  

def extract_and_rename_zips(zip_dir, directory):
    for filename in os.listdir(zip_dir):
        print(filename)
        if filename.endswith('.zip') and 'ravi_style_gt' in filename:
            zip_file_path = os.path.join(zip_dir, filename)
            folder_name = filename.split('_')[1]
            extract_directory = os.path.join(directory, 'newdata_', f'{folder_name}_ravi_style_gt')
            # Create the directory if it doesn't exist
            os.makedirs(extract_directory, exist_ok=True)
            try:
                # Use subprocess to execute the unzip command
                subprocess.run(['unzip', zip_file_path, '-d', extract_directory])
            except subprocess.CalledProcessError:
                print(f"Skipping {filename} - Not a valid zip file")


extract_and_rename_zips('/home/user/docTR/New_data', current_dir)

