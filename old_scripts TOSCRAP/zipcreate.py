import zipfile
import os
def create_zip(source_folder, zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, arcname=os.path.relpath(file_path, source_folder))

# Example usage
source_folder = '/data3/sreevatsa/Datasets/dataset2'
zip_file_path = '/data3/sreevatsa/Datasets/dataset2.zip'
create_zip(source_folder, zip_file_path)
