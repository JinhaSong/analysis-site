import json
import requests

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


def get_pot_segmentation(url, cls_result_data):
    seg_data = dict()
    seg_file = json.dumps(cls_result_data, ensure_ascii=False, indent='')
    seg_files = {'file': seg_file}
    seg_response = requests.post(url=url, data=seg_data, files=seg_files)
    seg_result_data = json.loads((seg_response.content).decode('utf-8'))
    seg_image = seg_result_data['result']

    return seg_image