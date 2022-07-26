import cv2
import os, sys
import json
from xml.dom.minidom import Document
from tqdm import tqdm
import argparse
from glob import glob
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description='VOC')
parser.add_argument('--path', type=str, help='path to data')

args = parser.parse_args()

anns = glob(os.path.join(args.path, 'Annotations/*xml'))
txt = open(os.path.join(args.path, 'ImageSets/Main/train.txt'), 'w')
count = 0
for ann in anns:
    os.rename(ann, ann.split('.')[0].replace('_leftImg8bit', '') + '_leftImg8bit.xml')
anns = glob(os.path.join(args.path, 'Annotations/*xml'))
for ann in anns:
    tree = ET.parse(ann)
    root = tree.getroot()
    num = 0
    for obj in root.iter('object'):
        num += 1
    if num == 0:
        continue
    num = 0
    file = os.path.basename(ann).split('.')[0]
    # assert os.path.isfile(os.path.join(args.path, 'JPEGImages/{}.png'.format(file)))
    txt.write(file + '\n')
    count += 1

print('{}/{}'.format(count, len(anns)))