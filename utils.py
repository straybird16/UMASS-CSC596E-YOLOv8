import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

import json
import shutil
import random
import os
import numpy as np

# Function to get the data from XML Annotation
def extract_info_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    
    # Initialise the info dict 
    info_dict = {}
    info_dict['bboxes'] = []

    # Parse the XML Tree
    for elem in root:
        # Get the file name 
        if elem.tag == "filename":
            info_dict['filename'] = elem.text
            
        # Get the image size
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))
            
            info_dict['image_size'] = tuple(image_size)
        
        # Get details of the bounding box 
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text
                    
                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)            
            info_dict['bboxes'].append(bbox)
    
    return info_dict

# Convert the info dict to the required yolo format and write it to disk
def convert_to_yolov5(info_dict, path_annotations:str='temp_data/annotations'):
    # Dictionary that maps class names to IDs
    class_name_to_id_mapping = {"trafficlight": 0,
                           "stop": 1,
                           "speedlimit": 2,
                           "crosswalk": 3}
    
    print_buffer = []
    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
        
        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2 
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])
        
        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]  
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        
        #Write the bbox details to the file 
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
        
    # Name of the file which we have to save 
    save_file_name = os.path.join(path_annotations, info_dict["filename"].replace("png", "txt"))
    
    # Save the annotation to disk
    print("\n".join(print_buffer), file= open(save_file_name, "w"))


def plot_bounding_box(image, annotation_list):
    class_name_to_id_mapping = {"trafficlight": 0,
                           "stop": 1,
                           "speedlimit": 2,
                           "crosswalk": 3}
    class_id_to_name_mapping = dict(zip(class_name_to_id_mapping.values(), class_name_to_id_mapping.keys()))
    annotations = np.array(annotation_list)
    w, h = image.size
    
    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
    transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h 
    
    transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
    transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
    transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
    transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]
    
    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0,y0), (x1,y1)))
        
        plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))])
    
    plt.imshow(np.array(image))
    plt.show()


def partition_dataset(path_images:str='temp_data/images', path_annotations:str='temp_data/annotations'):
    # Read images and annotations into 2 lists and sort
    images = [os.path.join(path_images, x) for x in os.listdir(path_images)]
    annotations = [os.path.join(path_annotations, x) for x in os.listdir(path_annotations) if x[-3:] == "txt"]
    images.sort()
    annotations.sort()
    # Split the dataset into train-validation-test splits 
    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
    val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)

    def move_files_to_folder(list_of_files:list, destination_folder:str):
        if not os.path.isdir(destination_folder):
            os.mkdir(destination_folder)
        for f in list_of_files:
            try:
                filename = os.path.split(f)[1]
                #print("destination folder: ", destination_folder)
                #print("final path: ", destination_folder+filename)
                os.replace(f, os.path.join(destination_folder, filename))
                #shutil.move(f, destination_folder+filename)
            except:
                print(f)
                assert False

    move_files_to_folder(train_images, os.path.join(path_images, 'train'))
    print("Moved train images to ", os.path.join(path_images, 'train'))

    move_files_to_folder(val_images, os.path.join(path_images, 'val'))
    print("Moved val images to ", os.path.join(path_images, 'val'))
    
    move_files_to_folder(test_images, os.path.join(path_images, 'test'))
    print("Moved test images to ", os.path.join(path_images, 'test'))

    move_files_to_folder(train_annotations, os.path.join(path_annotations, 'train'))
    print("Moved train annotations to ", os.path.join(path_annotations, 'train'))

    move_files_to_folder(val_annotations, os.path.join(path_annotations, 'val'))
    print("Moved val annotations to ", os.path.join(path_annotations, 'val'))

    move_files_to_folder(test_annotations, os.path.join(path_annotations, 'test'))
    print("Moved test annotations to ", os.path.join(path_annotations, 'test'))

def myTrain(model, data:str='temp_data/temp_data.yaml', hyp:str='temp_data/hyp.txt'):
    args = json.loads(open(hyp).read())
    args['data'] = data
    return model.train(**args)
