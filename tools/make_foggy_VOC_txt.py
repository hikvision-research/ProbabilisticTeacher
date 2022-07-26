import cv2
import os, sys
import json
from xml.dom.minidom import Document
from tqdm import tqdm
import argparse
from glob import glob
import xml.etree.ElementTree as ET
import shutil

parser = argparse.ArgumentParser(description='VOC')
parser.add_argument('--path', type=str, help='path to data')

args = parser.parse_args()

anns = glob(os.path.join(args.path, 'Annotations/*xml'))
txt = open(os.path.join(args.path, 'ImageSets/Main/val.txt'), 'w')
count = 0
for ann in anns:
    shutil.copyfile(ann, ann.split('.')[0].replace('_foggy_beta_0.005', '') + '_leftImg8bit_foggy_beta_0.005.xml')
    shutil.copyfile(ann, ann.split('.')[0].replace('_foggy_beta_0.01', '') + '_leftImg8bit_foggy_beta_0.01.xml')
    shutil.copyfile(ann, ann.split('.')[0].replace('_foggy_beta_0.02', '') + '_leftImg8bit_foggy_beta_0.02.xml')
    os.remove(ann)
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
    file = os.path.basename(ann).replace('.xml', '')
    assert os.path.isfile(os.path.join(args.path, 'JPEGImages/{}.png'.format(file)))
    txt.write(file + '\n')
    count += 1

print('{}/{}'.format(count, len(anns)))