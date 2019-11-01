from __future__ import print_function

from AnalysisSite.celerys import app
import json
import requests
import ast
import time


@app.task
def communicator(url, image_path):
    start = time.time()
    cls_data = dict()
    cls_image = open(image_path, 'rb')
    cls_files = {'image': cls_image}
    cls_response = requests.post(url=url, data=cls_data, files=cls_files)
    cls_result_data = json.loads((cls_response.content).decode('utf-8'))
    end = time.time()
    print("====== classification fin {} ======".format(end-start))

    start = time.time()
    seg_data = dict()
    seg_file = json.dumps(cls_result_data, ensure_ascii=False, indent='')
    seg_files = {'file': seg_file}
    seg_response = requests.post(url='http://mltwins.sogang.ac.kr:8002', data=seg_data, files=seg_files)
    end = time.time()
    print("====== segmentation fin {} ======".format(end-start))

    seg_result_data = json.loads((seg_response.content).decode('utf-8'))
    seg_image = seg_result_data['result'][0]['label'][0]['description']

    start = time.time()
    cls_detail_data = dict()
    cls_detail_data['image'] = seg_image
    cls_detail_response = requests.post(url='http://mltwins.sogang.ac.kr:8003', data=cls_detail_data)
    cls_detail_result_data = json.loads((cls_detail_response.content).decode('utf-8'))
    end = time.time()
    print("====== classification detail fin {} ======".format(end-start))

    for i in range(len(cls_result_data['result'])) :
        cls_position = cls_result_data['result'][i]['position']
        for j in range(len(cls_detail_result_data['result'])) :
            cls_detail_position = cls_detail_result_data['result'][j]['position']
            if cls_position['x'] == cls_detail_position['x'] and cls_position['y'] == cls_detail_position['y'] :
                cls_result_data['result'][i]['label'].extend(cls_detail_result_data['result'][j]['label'])
                j = len(cls_detail_result_data['result'])
                print(i, j, cls_position['x'], cls_detail_position['x'], cls_position['y'], cls_detail_position['y'])

    cls_image.close()

    result = {}
    result['classification_result'] = cls_result_data['result']
    result['image'] = seg_image

    return result
