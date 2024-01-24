#convert jpg png images to tiff images

#also catches those files that throw DecompressionBombWarnings and prints them into a txt file

from PIL import Image
import os
import warnings

input_folder = '/data3/sreevatsa/Datasets/exp/'
output_folder = '/data3/sreevatsa/Datasets/exp/dataset5_8124/tiff_output'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

for root, _, files in os.walk(input_folder):
    for filename in files:
        # Check if the file has the desired extension (e.g., '.jpg')
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Open the image
            try:
                print("Processing file: {}".format(os.path.join(root, filename)))
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    with Image.open(os.path.join(root, filename)) as img:
                        # If a warning has been triggered
                        if len(w) > 0 and issubclass(w[-1].category, Image.DecompressionBombWarning):
                            print(f"DecompressionBombWarning for file: {os.path.join(root, filename)}",file=open("/data3/sreevatsa/Datasets/rr.txt","a"))
                        
                        # Construct the output file path with '.tiff' extension
                        output_path = os.path.join(output_folder, os.path.relpath(root, input_folder), os.path.splitext(filename)[0] + '.tiff')
                        
                        # Ensure the subfolder structure exists in the output path
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        
                        # Convert and save as TIFF format
                        img.save(output_path, format='TIFF', compression='tiff_lzw')
            except Exception as e:
                print("Error converting file: {}".format(os.path.join(root, filename)),file=open("/data3/sreevatsa/Datasets/rr.txt","a"))
                print(e)        
print("Conversion complete.")
