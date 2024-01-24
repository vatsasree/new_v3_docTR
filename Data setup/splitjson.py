import json
import os

# def split_json(input_file):
#     with open(input_file, 'r') as json_file:
#         data = json.load(json_file)
        
#         for image_path, bounding_boxes in data.items():
#             #print(image_path, (bounding_boxes))
            
#             fname = f"{os.path.splitext(os.path.basename(image_path))[0]}.json"
#             output_file = os.path.join("/data3/sreevatsa/mergedData/split_jsons/",fname)
#             with open(output_file, 'w') as output_json:
#                 json.dump({image_path: bounding_boxes}, output_json, indent=2)
#             print(f"Created {output_file}")

def split_json(input_file):
    with open(input_file, 'r') as json_file:
        data = json.load(json_file)
        print("TOTAL len:",len(data.keys()))
        c=0
        for image_path, bounding_boxes in data.items():
            #print(image_path, (bounding_boxes))
            try:
                fname = f"{os.path.splitext(os.path.basename(image_path))[0]}.json"
                output_dir = "/scratch/sreevatsa/mergedData2/split_jsons/"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_file = os.path.join(output_dir, fname)
                with open(output_file, 'w') as output_json:
                    json.dump({image_path: bounding_boxes}, output_json, indent=2)
                print(f"Created {output_file}")
                c+=1
                print("Created: ",c)
            except:
                print("Error in creating the file for: ",image_path,file=open('/home2/sreevatsa/rrr.txt','w'))
                continue

# Replace 'your_input_file.json' with the actual path to your input JSON file
split_json('/scratch/sreevatsa/mergedData2/mergedJson.json')


