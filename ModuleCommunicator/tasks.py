from __future__ import print_function

from AnalysisSite.celerys import app
import json
import requests
import ast
import time

@app.task
def communicator(url, image_path, cc_th, severity_th):
    start = time.time()
    cls_data = dict()
    cls_image = open(image_path, 'rb')
    cls_files = {'image': cls_image}
    cls_response = requests.post(url=url, data=cls_data, files=cls_files)
    cls_result_data = json.loads((cls_response.content).decode('utf-8'))
    cls_image.close()
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

    start = time.time()
    region_result = make_region(cls_result_data)
    end = time.time()
    print("====== region fin {} ======".format(end - start))

    for i in range(len(cls_result_data['result'])) :
        cls_position = cls_result_data['result'][i]['position']
        for j in range(len(cls_detail_result_data['result'])) :
            cls_detail_position = cls_detail_result_data['result'][j]['position']
            if cls_position['x'] == cls_detail_position['x'] and cls_position['y'] == cls_detail_position['y'] :
                cls_result_data['result'][i]['label'].extend(cls_detail_result_data['result'][j]['label'])
                j = len(cls_detail_result_data['result'])

    result = {}
    result['classification_result'] = cls_result_data['result']
    result['seg_image'] = seg_image
    result['region_result'] = region_result
    return result


def make_region(cls_result_data):
    CLASS_LIST = ['normal', 'crack', 'patch', 'lane', 'detail_norm', 'ac', 'tc', 'lc']
    CLASS_NUM = len(CLASS_LIST)
    image_infos = cls_result_data['result']

    # image info
    # TODO: not to use static size
    image_width = 3704
    image_height = 10000
    patch_size = 256

    axis_x = int(image_width / patch_size)
    axis_y = int(image_height / patch_size)
    crack_map = [[0 for x in range(axis_x)] for y in range(axis_y)]

    # Scan all result data
    for image_info in image_infos:
        x = int(image_info['position']['x'] / image_info['position']['w'])
        y = int(image_info['position']['y'] / image_info['position']['h'])
        label = image_info['label']

        classification = {'normal': 0, 'crack': 0, 'patch': 0, 'lane': 0}
        crack_classification = {'detail_norm': 0, 'ac': 0, 'tc': 0, 'lc': 0}

        for l in label:
            if l['description'] in classification:
                classification[l['description']] = l['score']
            elif l['description'] in crack_classification:
                crack_classification[l['description']] = l['score']

        classification_max = max(classification.keys(), key=(lambda k: classification[k]))
        crack_classification_max = max(crack_classification.keys(), key=(lambda k: crack_classification[k]))

        # Make 2D crack map
        if classification_max == 'patch':
            crack_map[y][x] = 2
        elif classification_max == 'crack':
            if crack_classification_max == 'detail_norm':
                crack_map[y][x] = 0
            elif crack_classification_max == 'ac':
                crack_map[y][x] = 5
            elif crack_classification_max == 'tc':
                crack_map[y][x] = 6
            elif crack_classification_max == 'lc':
                crack_map[y][x] = 7

    check_pos_list = [-1, 0, 1]
    num_of_different_region = [0 for x in range(CLASS_NUM)]

    # Region proposing algorithm
    for current_x in range(0, axis_x):
        for current_y in range(0, axis_y):
            is_connected_component_exist = False
            curernt_value = crack_map[current_y][current_x]

            # if crackmap is not normal or not lane
            if curernt_value > 0:
                for check_pos_add_to_x in check_pos_list:
                    for check_pos_add_to_y in check_pos_list:
                        check_pos_x = current_x + check_pos_add_to_x
                        check_pos_y = current_y + check_pos_add_to_y
                        # Not out of index
                        if check_pos_x >= 0 and check_pos_x < axis_x and check_pos_y >= 0 and check_pos_y < axis_y:
                            check_pos_value = crack_map[check_pos_y][check_pos_x]
                            # if checking pos has region name, than change curren pos region name
                            if int(check_pos_value / 100) == curernt_value:
                                crack_map[current_y][current_x] = check_pos_value
                                is_connected_component_exist = True
                if is_connected_component_exist == False:
                    crack_map[current_y][current_x] = curernt_value * 100 + num_of_different_region[curernt_value]
                    num_of_different_region[curernt_value] += 1

    # remove duplicate region and make json
    result = []
    for current_x in range(0, axis_x):
        for current_y in range(0, axis_y):
            curernt_value = crack_map[current_y][current_x]
            if curernt_value > 0:
                for check_pos_add_to_x in check_pos_list:
                    for check_pos_add_to_y in check_pos_list:
                        check_pos_x = current_x + check_pos_add_to_x
                        check_pos_y = current_y + check_pos_add_to_y

                        if check_pos_x >= 0 and check_pos_x < axis_x and check_pos_y >= 0 and check_pos_y < axis_y:
                            check_pos_value = crack_map[check_pos_y][check_pos_x]
                            if int(check_pos_value / 100) == int(curernt_value / 100) and curernt_value > 10:
                                for change_x in range(0, axis_x):
                                    for change_y in range(0, axis_y):
                                        if crack_map[change_y][change_x] == curernt_value:
                                            crack_map[change_y][change_x] = check_pos_value
    # make result
    for current_x in range(0, axis_x):
        for current_y in range(0, axis_y):
            curernt_value = crack_map[current_y][current_x]
            if curernt_value > 0:
                is_json_region_not_exist = True
                if len(result) > 0:
                    for i in range(0, len(result)):
                        if result[i]['region'] == curernt_value:
                            is_json_region_not_exist = False
                            result[i]['region_area'].append({
                                'h': patch_size, 'w': patch_size,
                                'x': current_x * patch_size, 'y': current_y * patch_size
                            })

                if len(result) == 0 or is_json_region_not_exist:
                    result.append({
                        'region': curernt_value,
                        'region_type': CLASS_LIST[int(curernt_value / 100)],
                        'region_area': [{
                            'h': patch_size, 'w': patch_size,
                            'x': current_x * patch_size, 'y': current_y * patch_size
                        }]
                    })
    for i in range(0, len(result)):
        result[i]['region'] = i

    return result