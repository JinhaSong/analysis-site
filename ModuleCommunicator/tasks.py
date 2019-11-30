from __future__ import print_function

from AnalysisSite.celerys import app

import ast
import time
from ModuleCommunicator.utils.region import make_region
from ModuleCommunicator.utils.severity import crack_width_analysis
from ModuleCommunicator.utils.module_result import *
from ModuleCommunicator.utils.image import *

from cv2 import cv2
import numpy as np
from PIL import Image
import math
import glob
import os

@app.task
def communicator(url, image_path, image_width, image_height, patch_size, region_thresold, region_connectivity, region_noise_filter, severity_threshold):
    url_cls = 'http://mltwins.sogang.ac.kr:8001'
    url_seg = 'http://mltwins.sogang.ac.kr:8002'
    url_cls_detail = 'http://mltwins.sogang.ac.kr:8003'
    url_seg_pot = 'http://mltwins.sogang.ac.kr:8004'

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
    seg_image_pot = get_pot_segmentation(url_seg_pot, cls_result_data)
    end = time.time()
    print("====== pot segmentation fin {} ======".format(end - start))

    classification_result = cls_result_data
    cls_result_data = cls_result_data['results'][0]['module_result']

    start = time.time()
    severity_result = crack_width_analysis(seg_image, severity_threshold, cls_result_data, patch_size=256)
    end = time.time()
    print("====== severity fin {} ======".format(end - start))

    start = time.time()
    for i in range(len(cls_result_data)) :
        cls_position = cls_result_data[i]['position']
        cls_detail_result = cls_detail_result_data['results'][0]['module_result']
        for j in range(len(cls_detail_result)) :
            cls_detail_position = cls_detail_result[j]['position']
            if cls_position['x'] == cls_detail_position['x'] and cls_position['y'] == cls_detail_position['y'] :
                cls_result_data[i]['label'].extend(cls_detail_result[j]['label'])
                j = len(cls_detail_result)

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
    end = time.time()
    print("====== json data parse fin {} ======".format(end - start))

    start = time.time()
    region_results = make_region(image_path, classification_result, image_width, image_height, patch_size, region_thresold, connectivity_option=region_connectivity, noise_filtering_option=region_noise_filter)
    end = time.time()
    print("====== region fin {} ======".format(end - start))

    start = time.time()
    seg_full_image = attach_pot(seg_image, seg_image_pot)
    end = time.time()
    print("====== attach pothole image fin {} ======".format(end - start))

    start = time.time()
    seg_full_image = attach_patch(region_results, severity_threshold, seg_full_image)
    end = time.time()
    print("====== attach patch image fin {} ======".format(end - start))

    start = time.time()
    result_image = make_result_image(region_results, severity_threshold, seg_full_image)
    end = time.time()
    print("====== create result image fin {} ======".format(end - start))



    result = {}
    result['cls_result'] = cls_result_data
    result['seg_image'] = convert_image_binary(seg_full_image)
    result['region_result'] = region_results
    result['result_image'] = convert_image_binary(result_image)
    return result



