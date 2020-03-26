import xml.etree.ElementTree as ET
import csv
import pandas as pd

'''
This python script creates a csv file with the data from a ground truth xml file:
framenumber, objectID, ul_x, ul_y, width, height
'''
tree = ET.parse("GroundTruths_xml/wk1gt.xml")
root = tree.getroot()

# open a file for writing
ground_truth = open('GroundTruths_csv/wk1gt.csv', 'w')

# create the csv writer object
csvwriter = csv.writer(ground_truth)
frame_headers = []

count = 0
for frame in root.findall('frame'):
    frame_write = []
    frame_nr = frame.attrib.get('number')
    frame_write.append(frame_nr)

    for object in frame[0].findall('object'):
        frame_write.append(object.attrib.get('id'))
        box = object.find('box')
        frame_write.append(box.attrib.get('xc'))
        frame_write.append(box.attrib.get('yc'))
        frame_write.append(box.attrib.get('w'))
        frame_write.append(box.attrib.get('h'))

    csvwriter.writerow(frame_write)
ground_truth.close()
