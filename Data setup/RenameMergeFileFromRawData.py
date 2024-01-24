import os
import json
import shutil

def rename_images(directory):
    print(directory)
    parent_folder_id = directory.split('_')[1]
    parent_folder_id = parent_folder_id.replace('/', '')
    print(parent_folder_id)
    for filename in os.listdir(directory):
        if filename.endswith((".jpg", ".jpeg", ".png",".tif",".tiff",".bmp")):
            img_id, ext = os.path.splitext(filename)
            new_img_name = f'{parent_folder_id}_{img_id}{ext}'
        #    os.rename(os.path.join(directory, filename), os.path.join(directory, new_img_name))
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_img_name)
            shutil.move(old_file_path, new_file_path)
    return parent_folder_id

def update_json(json_dir,parent_folder_id):
    with open(json_dir) as json_file:
        data_dict = json.load(json_file)
    parent_folder_id = directory.split('_')[1]
    parent_folder_id = parent_folder_id.replace('\\', '')
    keys = data_dict.keys()
    # for key in keys:
    #     print(key)
    prefixed_data_dict = {}
    prefix = f"{parent_folder_id}_"
    for key, value in data_dict.items():
        if not key.startswith(prefix):
            prefixed_key = f'{prefix}{key}'
        else:
            prefixed_key = key
        prefixed_data_dict[prefixed_key] = value

    with open(json_dir, 'w') as json_file:
        json.dump(prefixed_data_dict, json_file, indent=4)



def merge_images_and_json(source_folder, destination_folder, destination_json):
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    merged_json_data = {}
    
    for folder in os.listdir(source_folder):
        folder_path = os.path.join(source_folder, folder)
        print("Folder:",folder," Folder path:",folder_path)
        if os.path.isdir(folder_path):
            for root, _, filenames in os.walk(folder_path):
                print('Filenames list is as',filenames)
                for filename in filenames:
                    print("Filename:",filename)
                    file_path = os.path.join(root, filename)
                    print("File path:",file_path)

                    if filename.endswith((".jpg", ".jpeg", ".png",".tif",".tiff",".bmp")):
                        shutil.copy(file_path, destination_folder)
                        print("Copied image:", file_path, "to ", destination_folder)
                    if filename.endswith(".json"):
                        with open(file_path) as json_file:
                            json_data = json.load(json_file)

                        merged_json_data.update(json_data)
    with open(destination_json, "w") as json_file:
        json.dump(merged_json_data, json_file, indent=4)

    print("Images and JSON files merged successfully.")


if __name__ == "__main__":
    # source_folder = '/home/ndli19/docvisor/UseCaseDocsData/unzips'
    # count=0
    # for folder in os.listdir(source_folder):
    #     count+=1
    #     print("COUNT:",count)
    #     folder_path = os.path.join(source_folder, folder)
    #     print(folder_path)
    #     # Check if it's a directory
    #     if os.path.isdir(folder_path):
    #         # Recursively traverse all files and subfolders in the folder
    #         folder_path = os.path.join(folder_path,'ravi_style_gt')
    #         # print('Current folder path',folder_path)
    #         directory = os.path.join(folder_path,'images')
    #         try:
    #             parent_folder_id = rename_images(directory)
    #             for root,_, filenames in os.walk(folder_path):
    #                 # print('Filenames list is as',filenames)
    #                     # Check if it's a JSON file
    #                 for filename in filenames:
    #                     if filename.endswith(".json"):
    #                         json_dir = os.path.join(folder_path,filename)
    #                         update_json(json_dir,parent_folder_id)
    #         except:
    #             print(directory,file=open("error.txt", "a"))
    #             # print('No images in the folder')
    merge_images_and_json("/home/ndli19/docvisor/UseCaseDocsData/unzips","/home/ndli19/docvisor/UseCaseDocsData/mergedData/mergedImages", "/home/ndli19/docvisor/UseCaseDocsData/mergedData/merged_newdata2_json.json")
