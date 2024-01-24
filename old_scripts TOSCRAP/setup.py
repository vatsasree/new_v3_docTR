import glob
import shutil
import os
import json
from PIL import Image
import hashlib
import argparse
import random
import math
parser = argparse.ArgumentParser()

parser.add_argument("--pathj",
                    help="Path to json",
                    type=str, default="/data3/sreevatsa/reclists/image_recall_list.json")
parser.add_argument("--paths",
                    help="Path to save files",
                    type=str, default="/data3/sreevatsa/Datasets/dataset3")
parser.add_argument("--pathg",
                    help="Path to ground truth",
                    type=str, default="/data3/sreevatsa/merged2/split_jsons/")
parser.add_argument("--ext",
                    help="Extension of files to be considered (supports one at this moment)",
                    type=str, default='jpg')
parser.add_argument("--cutoffs",
                    help='list of two cutoffs for dividing into easy, medium, hard',
                    nargs='+', type=float, default=[10, 40])
#0.63 0.45
parser.add_argument("--languages",
                    help='list of languages in dataset/to be used in current run',
                    nargs='+', type=str, default=["Assamese", "Bangla", "Gujarati", "Gurumukhi", "Hindi", "Kannada", "Malayalam", "Manipuri", "Marathi", "Oriya", "Tamil", "Telugu", "Urdu"])
parser.add_argument("--region",
                    help="Region of interest (line or word)",
                    type=str, default='word')
parser.add_argument("--split",
                    help='train val test split percents',
                    nargs='+', type=float, default=[60, 20, 20])
args = parser.parse_args()


# extension of image to be considered
ext = args.ext
# paths to different folders
# path_to_dataset = args.pathd
path_to_json = args.pathj
path_to_groundtruth = args.pathg
path_to_save = args.paths
# iou values to consider
cutoffs = args.cutoffs
# languages under consideration for current run
languages = args.languages
# reg under consideration. Either word or line
regiondecision = args.region
split = args.split
# loading all ground truths for languages
print('Loading ground truths....')
data = {}
# for language in languages:
#     path_to_groundtruth = "GT/docvisor_consortium_gt/"+language + ".json"
#     f = open(path_to_groundtruth)
#     data[language] = json.load(f)

# f = open(path_to_groundtruth)
# data = json.load(f)
# print(data.keys())
# makes the labels for training


def make_labels(impath, labels):
    img = Image.open(impath)

    # get width and height
    width = img.width
    height = img.height

    dimensions = img.size

    # display width and height
    # print("dimensions: ",dimensions)
    # print("The height of the image is: ", height)
    # print("The width of the image is: ", width)

    readable_hash = ""
    with open(impath, "rb") as f:
        bytes = f.read()  # read entire file as bytes
        readable_hash = hashlib.sha256(bytes).hexdigest()
        # print(type(readable_hash))

    ans_key = ""

    test_img_path = os.path.basename(impath)

    split_direc = '/data3/sreevatsa/merged2/split_jsons'
    jsonname = f"{os.path.splitext(os.path.basename(test_img_path))[0]}.json"
    jsondirec = os.path.join(split_direc, jsonname)

    ff=open(jsondirec)
    datasplit=json.load(ff)
    ff.close()

    image_keys = list(datasplit.keys())

    for k in image_keys:
        # path = os.path.basename(data[k]["imagePath"])
        path = k.split('/')[1]
        if path == test_img_path:
            ans_key = k
            break

    # if ans_key != "":
    #     break

    if ans_key == "":
        print('something is wrong1')

    # regions = []
    # for i, region in enumerate(data[ans_key]["regions"]):
    #     if region["regionLabel"] == regiondecision:
    #         regions.append(region["groundTruth"])
    # # print(regions)
    regions=[]
    for i in datasplit.keys():
    # print(al[i]['words'])
        for ii in datasplit[i]['words']:
            x1,y1,x2,y2 = ii['groundTruth']
            aa=[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            regions.append(aa)

    # print(regions)
    
    labels[test_img_path] = {
        'img_dimensions': dimensions,
        'img_hash': readable_hash,
        'polygons': regions
    }


f = open(path_to_json)
all_recalls_docs = json.load(f)
f.close()

# docs = [element[2] for element in all_recalls_docs] #saving the file names in docs unnecessarily when we can directly use the file names from all_recalls_docs

print('making the files and folders...')
if os.path.isdir(path_to_save):
    shutil.rmtree(path_to_save)

os.mkdir(path_to_save)
os.mkdir(path_to_save + '/train/')
os.mkdir(path_to_save + '/val/')
os.mkdir(path_to_save + '/val/images')
os.mkdir(path_to_save + '/test/')
os.mkdir(path_to_save + '/test/images')
os.mkdir(path_to_save + '/train/Easy/')
os.mkdir(path_to_save + '/train/Medium/')
os.mkdir(path_to_save + '/train/Hard/')
os.mkdir(path_to_save + '/train/Easy/images')
os.mkdir(path_to_save + '/train/Medium/images')
os.mkdir(path_to_save + '/train/Hard/images')


cut_off_1 = 0.675
cut_off_2 = 0.85
print('splitting into easy, medium, hard...')
print('cutoffs: ', cut_off_1, cut_off_2)
# print([(item[0]) for item in all_recalls_docs])
# easylist = [item[2] for item in data if float(item[0]) > cut_off_2]
# mediumlist = [item[2] for item in data if float(
#     item[0]) > cut_off_1 and float(item[0]) <= cut_off_2]
# hardlist = [item[2] for item in data if float(item[0]) <= cut_off_1]

# e1=[item[0] for item in all_recalls_docs if float(item[0]) > cut_off_2]
# e2=[item[0] for item in all_recalls_docs if float(item[0]) > cut_off_1 and float(item[0]) <= cut_off_2]
# e3=[item[0] for item in all_recalls_docs if float(item[0]) <= cut_off_1]

# print(e3)
# print(len(e1),len(e2),len(e3))

easylist = [os.path.join("/data3/sreevatsa/merged2/merged_newdata2",item[2]) for item in all_recalls_docs if float(item[0]) > cut_off_2]
mediumlist = [os.path.join("/data3/sreevatsa/merged2/merged_newdata2",item[2]) for item in all_recalls_docs if float(
    item[0]) > cut_off_1 and float(item[0]) <= cut_off_2]
hardlist = [os.path.join("/data3/sreevatsa/merged2/merged_newdata2",item[2]) for item in all_recalls_docs if float(item[0]) <= cut_off_1]

# easylist = docs[:cutoffs[0]]
# mediumlist = docs[cutoffs[0]:cutoffs[1]]
# hardlist = docs[cutoffs[1]:]

random.shuffle(easylist)
random.shuffle(mediumlist)
random.shuffle(hardlist)

traineasy = easylist[:math.floor(len(easylist)*split[0]/100)]
val = easylist[math.floor(len(easylist)*split[0]/100)
                          :math.floor(len(easylist)*(split[0]+split[1])/100)]
test = easylist[math.floor(len(easylist)*(split[0]+split[1])/100):]

trainmedium = mediumlist[:math.floor(len(mediumlist)*split[0]/100)]
val = val + mediumlist[math.floor(len(mediumlist)*split[0]/100)
                                  :math.floor(len(mediumlist)*(split[0]+split[1])/100)]
test = test + mediumlist[math.floor(len(mediumlist)*(split[0]+split[1])/100):]

trainhard = hardlist[:math.floor(len(hardlist)*split[0]/100)]
val = val + hardlist[math.floor(len(hardlist)*split[0]/100)
                                :math.floor(len(hardlist)*(split[0]+split[1])/100)]
test = test + hardlist[math.floor(len(hardlist)*(split[0]+split[1])/100):]

# print(traineasy) #add full directory path to the file name is all the above lists, currently only file name is there
print("starting train hard")

# labels = {}
# chunk_size = 100

# for i in range(0, len(traineasy), chunk_size):
#     for tiffile in traineasy[i:i+chunk_size]:
#         shutil.copy(tiffile, path_to_save + '/train/Hard/images')
#         make_labels(tiffile, labels)

#     with open(path_to_save + '/train/Hard/labels.json', "a") as outfile:
#         json.dump(labels, outfile, indent=6)
#   # Reset the labels dictionary

labels = {}
for i,tiffile in enumerate(traineasy):
    print(i)
    shutil.copy(tiffile, path_to_save + '/train/Hard/images')
    make_labels(tiffile, labels)

outfile = open(path_to_save + '/train/Hard/labels.json', "w")
json.dump(labels, outfile, indent=6)
outfile.close()

print("starting train medium")
labels = {}
for i,tiffile in enumerate(trainmedium):
    print(i)
    shutil.copy(tiffile, path_to_save + '/train/Medium/images')
    make_labels(tiffile, labels)

outfile = open(path_to_save + '/train/Medium/labels.json', "w")
json.dump(labels, outfile, indent=6)
outfile.close()

print("starting train easy")
labels = {}
for i,tiffile in enumerate(trainhard):
    print(i)
    shutil.copy(tiffile, path_to_save + '/train/Easy/images')
    make_labels(tiffile, labels)

outfile = open(path_to_save + '/train/Easy/labels.json', "w")
json.dump(labels, outfile, indent=6)
outfile.close()

print("starting val")
labels = {}
for tiffile in val:
    shutil.copy(tiffile, path_to_save + '/val/images')
    make_labels(tiffile, labels)

outfile = open(path_to_save + '/val/labels.json', "w")
json.dump(labels, outfile, indent=6)
outfile.close()

print("starting test")
labels = {}
for tiffile in test:
    shutil.copy(tiffile, path_to_save + '/test/images')
    make_labels(tiffile, labels)

outfile = open(path_to_save + '/test/labels.json', "w")
json.dump(labels, outfile, indent=6)
outfile.close()
print('done!')