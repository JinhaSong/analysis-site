from __future__ import print_function

from AnalysisSite.celerys import app
import json
import requests
import ast
import time
from ModuleCommunicator.utils.region import make_region
from ModuleCommunicator.utils.severity import crack_width_analysis

from cv2 import cv2
import numpy as np
from PIL import Image
import math
import glob
import os

@app.task
def communicator(url, image_path, patch_size, region_connectivity, region_noise_filter, severity_threshold):
    url_cls = 'http://mltwins.sogang.ac.kr:8001'
    url_seg = 'http://mltwins.sogang.ac.kr:8002'
    url_cls_detail = 'http://mltwins.sogang.ac.kr:8003'

    start = time.time()
    cls_result_data = get_classification(url_cls, image_path)
    end = time.time()
    print("====== classification fin {} ======".format(end-start))

    start = time.time()
    seg_image = get_segmentation(url_seg, cls_result_data)
    end = time.time()
    print("====== segmentation fin {} ======".format(end-start))

    start = time.time()
    cls_detail_result_data = get_classification_detail(url_cls_detail, seg_image, cls_result_data)
    end = time.time()
    print("====== classification detail fin {} ======".format(end-start))

    start = time.time()
    print("region_noise_filter", region_connectivity, region_noise_filter)
    region_results = make_region(cls_result_data, connectivity_option=region_connectivity, noise_filter_option=region_noise_filter)
    end = time.time()
    print("====== region fin {} ======".format(end - start))

    cls_result_data = cls_result_data['results'][0]['module_result']

    start = time.time()
    severity_result, count = crack_width_analysis(seg_image, severity_threshold, cls_result_data, iter=30, patch_size=256)
    end = time.time()

    print("====== {} patches / severity_result fin {} ======".format(count, end - start))

    # detail = [
    #     {'description': 'ac', 'score': 0},
    #     {'description': 'lc', 'score': 0},
    #     {'description': 'detail_norm', 'score': 0},
    #     {'description': 'tc', 'score': 0}
    # ]

    # for region_result in region_results:
    #     region_type = region_result['region_type']
    #     if region_type == 'patch' :
    #         region_result['region_area'][0]['label'].extend(detail)

    for i in range(len(cls_result_data)) :
        cls_position = cls_result_data[i]['position']
        cls_detail_result = cls_detail_result_data['results'][0]['module_result']
        for j in range(len(cls_detail_result)) :
            cls_detail_position = cls_detail_result[j]['position']
            if cls_position['x'] == cls_detail_position['x'] and cls_position['y'] == cls_detail_position['y'] :
                cls_result_data[i]['label'].extend(cls_detail_result[j]['label'])
                j = len(cls_detail_result)

    count = 0
    for i in range(len(cls_result_data)) :
        cls_position = cls_result_data[i]['position']
        for j in range(len(severity_result)) :
            if cls_position['x'] == severity_result[j]['x'] and cls_position['y'] == severity_result[j]['y'] :
                cls_result_data[i]['severity'] = {}
                cls_result_data[i]['severity']['total_max_width'] = severity_result[j]['total_max_width']
                cls_result_data[i]['severity']['total_average_width'] = severity_result[j]['total_average_width']
                cls_result_data[i]['severity']['minx'] = float(severity_result[j]['minx'])
                cls_result_data[i]['severity']['miny'] = float(severity_result[j]['miny'])
                cls_result_data[i]['severity']['maxx'] = float(severity_result[j]['maxx'])
                cls_result_data[i]['severity']['maxy'] = float(severity_result[j]['maxy'])
                cls_result_data[i]['severity']['max_width_x'] = float(severity_result[j]['max_width_x'])
                cls_result_data[i]['severity']['max_width_y'] = float(severity_result[j]['max_width_y'])
                count += 1
    print("count", count)


    result = {}
    result['cls_result'] = cls_result_data
    result['seg_image'] = seg_image
    result['region_result'] = region_results
    return result

def get_classification(url, image_path) :
    cls_data = dict()
    cls_image = open(image_path, 'rb')
    cls_files = {'image': cls_image}
    cls_response = requests.post(url=url, data=cls_data, files=cls_files)
    cls_result_data = json.loads((cls_response.content).decode('utf-8'))
    cls_image.close()

    return cls_result_data

def get_segmentation(url, cls_result_data) :
    seg_data = dict()
    seg_file = json.dumps(cls_result_data, ensure_ascii=False, indent='')
    seg_files = {'file': seg_file}
    seg_response = requests.post(url=url, data=seg_data, files=seg_files)
    seg_result_data = json.loads((seg_response.content).decode('utf-8'))
    seg_image = seg_result_data['result']

    return seg_image

def get_classification_detail(url, seg_image, cls_result_data) :
    cls_detail_data = dict()
    seg_file = json.dumps(cls_result_data, ensure_ascii=False, indent='')
    seg_files = {'file': seg_file}
    cls_detail_data['image'] = seg_image
    cls_detail_response = requests.post(url=url, data=cls_detail_data, files=seg_files)
    cls_detail_result_data = json.loads((cls_detail_response.content).decode('utf-8'))

    return cls_detail_result_data

