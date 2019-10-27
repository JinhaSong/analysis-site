from __future__ import print_function

from AnalysisSite.celerys import app
import json
import requests
import ast
import time


@app.task
def communicator(url, image_path):
    cls_data = dict()
    cls_image = open(image_path, 'rb')
    cls_files = {'image': cls_image}

    start = time.time()
    cls_response = requests.post(url=url, data=cls_data, files=cls_files)
    result_data = json.loads((cls_response.content).decode('utf-8'))
    end = time.time()
    print("====== classification fin {} ======".format(end-start))

    start = time.time()
    seg_data = dict()
    seg_file = json.dumps(result_data, ensure_ascii=False, indent='')
    seg_files = {'file': seg_file}
    seg_response = requests.post(url='http://mltwins.sogang.ac.kr:8002', data=seg_data, files=seg_files)
    end = time.time()
    print("====== segmentation fin {} ======".format(end-start))

    seg_result_data = json.loads((seg_response.content).decode('utf-8'))
    seg_image = seg_result_data['result'][0]['label'][0]['description']
    result = {}
    result['classification_result'] = result_data['result']
    result['image'] = seg_image

    cls_image.close()

    return result
