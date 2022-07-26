import cv2
import os, sys
import json
from xml.dom.minidom import Document
from tqdm import tqdm

root = '/home/data/gtFine_trainvaltest/gtFine'
xml_save_path = '/home/data/gtFine_trainvaltest/gtFine_single_xml'
all_classes = ['car']
class_dict = {'car': 0}


def main():
    main_listing = os.listdir(root)
    for vid in main_listing:
        tmp_dir = os.path.join(root, vid)
        tmp_listing = os.listdir(tmp_dir)
        save_path = os.path.join(xml_save_path, vid)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print('processing --------- {} ---------'.format(vid))
        for tmp_vid in tqdm(tmp_listing):
            tmp_name = tmp_vid.split('.')[0]
            tmp_name = tmp_name[:-16]
            try:
                tmp_form = tmp_vid.split('.')[1]
            except:
                continue

            if (tmp_form != 'json'):
                continue

            tmp_file = open(tmp_dir + "/" + tmp_vid, 'r')
            tmp_json = json.loads(tmp_file.read())

            doc = Document()
            tmp_anno = doc.createElement('annotation')
            doc.appendChild(tmp_anno)

            tmp_folder = doc.createElement('folder')
            folder_text = doc.createTextNode('Cityscapes')
            tmp_folder.appendChild(folder_text)
            tmp_anno.appendChild(tmp_folder)

            tmp_filename = doc.createElement('filename')
            file_name = doc.createTextNode(tmp_name + '.png')
            tmp_filename.appendChild(file_name)
            tmp_anno.appendChild(tmp_filename)

            height = tmp_json['imgHeight']
            width = tmp_json['imgWidth']

            tmp_size = doc.createElement('size')
            tmp_depth = doc.createElement('depth')
            depth_text = doc.createTextNode('3')
            tmp_depth.appendChild(depth_text)
            tmp_size.appendChild(tmp_depth)
            tmp_width = doc.createElement('width')
            width_text = doc.createTextNode(str(width))
            tmp_width.appendChild(width_text)
            tmp_size.appendChild(tmp_width)
            tmp_height = doc.createElement('height')
            height_text = doc.createTextNode(str(height))
            tmp_height.appendChild(height_text)
            tmp_size.appendChild(tmp_height)
            tmp_anno.appendChild(tmp_size)

            tmp_obj = tmp_json['objects']
            for inst in tmp_obj:
                tmp_label = inst['label']

                if not (tmp_label in all_classes):
                    continue

                class_dict[tmp_label] = class_dict[tmp_label] + 1

                tmp_poly = inst['polygon']
                tmp_x = []
                tmp_y = []

                for point in tmp_poly:
                    tmp_x.append(point[0])
                    tmp_y.append(point[1])

                x1 = min(tmp_x)
                x2 = max(tmp_x)
                y1 = min(tmp_y)
                y2 = max(tmp_y)

                if (x1 <= 0):
                    x1 = 1
                if (x2 >= width):
                    x2 = width - 1
                if (y1 <= 0):
                    y1 = 1
                if (y2 >= height):
                    y2 = height - 1

                tmp_inst = doc.createElement('object')
                tmp_inst_name = doc.createElement('name')
                inst_name_text = doc.createTextNode(tmp_label)
                tmp_inst_name.appendChild(inst_name_text)
                tmp_inst.appendChild(tmp_inst_name)

                tmp_bndbox = doc.createElement('bndbox')
                tmp_x1 = doc.createElement('xmin')
                x1_text = doc.createTextNode(str(x1))
                tmp_x1.appendChild(x1_text)
                tmp_bndbox.appendChild(tmp_x1)
                tmp_y1 = doc.createElement('ymin')
                y1_text = doc.createTextNode(str(y1))
                tmp_y1.appendChild(y1_text)
                tmp_bndbox.appendChild(tmp_y1)
                tmp_x2 = doc.createElement('xmax')
                x2_text = doc.createTextNode(str(x2))
                tmp_x2.appendChild(x2_text)
                tmp_bndbox.appendChild(tmp_x2)
                tmp_y2 = doc.createElement('ymax')
                y2_text = doc.createTextNode(str(y2))
                tmp_y2.appendChild(y2_text)
                tmp_bndbox.appendChild(tmp_y2)
                tmp_inst.appendChild(tmp_bndbox)

                tmp_anno.appendChild(tmp_inst)

            tmp_xml = open(os.path.join(save_path, tmp_name + '.xml'), 'wb')
            tmp_xml.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    print(class_dict)


main()