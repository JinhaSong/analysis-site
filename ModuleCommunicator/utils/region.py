# import json
# import itertools
#
# CLASS_LIST = ['normal', 'crack', 'patch', 'lane', 'detail_norm', 'ac', 'tc','lc']
# CLASS_NUM = len(CLASS_LIST)
#
# OPTION_CONNECTIVITY_4 = 0
# OPTION_CONNECTIVITY_8 = 1
#
# OPTION_NOISE_FILTER_NOTHING = 0
# OPTION_NOISE_FILTER_FILL_IN_THE_BLANK = 1
# OPTION_NOISE_FILTER_CUT_EDGE = 2
# OPTION_NOISE_FILTER_CUT_EDGE_AND_FILL_IN_THE_BLANK = 3
# CHECK_POSITION_LIST = [-1, 0 ,1]
#
# def make_region(cls_result_data, image_width=3704, image_height=10000, patch_size=256, connected_component_threshold=1, connectivity_option=1, noise_filtering_option=3):
#     results = cls_result_data['results']
#
#     for result in results:
#         image_infos = result['module_result']
#
#         patch_size = 256
#         image_width = 3704
#         image_height = 10000
#
#         axis_x = int(image_width / patch_size)
#         axis_y = int(image_height / patch_size)
#         crack_map = [[0 for x in range(axis_x)] for y in range(axis_y)]
#
#         # Scan all result data
#         for image_info in image_infos:
#             x = int(image_info['position']['x'] / image_info['position']['w'])
#             y = int(image_info['position']['y'] / image_info['position']['h'])
#             label = image_info['label']




import json
import itertools
import base64
from PIL import Image
import requests
from io import BytesIO
import cv2
import numpy as np


CLASS_LIST = ['normal', 'crack', 'patch', 'lane', 'detail_norm', 'ac', 'tc','lc']
CLASS_NUM = len(CLASS_LIST)

OPTION_CONNECTIVITY_4 = 0
OPTION_CONNECTIVITY_8 = 1

OPTION_NOISE_FILTER_NOTHING = 0
OPTION_NOISE_FILTER_FILL_IN_THE_BLANK = 1
OPTION_NOISE_FILTER_CUT_EDGE = 2
OPTION_NOISE_FILTER_CUT_EDGE_AND_FILL_IN_THE_BLANK = 3
CHECK_POSITION_LIST = [-1, 0 ,1]

def make_region(image_path, cls_result_data, image_width=3704, image_height=10000, patch_size=256, region_threshold=0, connectivity_option=1, noise_filtering_option=3):
    results = cls_result_data['results']

    input_image = Image.open(image_path)

    # patch_size = 256
    # image_width = 3704
    # image_height = 10000

    patch_size = int(patch_size)
    image_width = int(image_width)
    image_height = int(image_height)

    for result in results:
        image_infos = result['module_result']

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
                    description = l['description']
                    # Remove detail_norm
                    if description == 'detail_norm':
                        crack_classification[description] = 0
                    else:
                        crack_classification[description] = l['score']

            classification_max = max(classification.keys(), key=(lambda k: classification[k]))
            crack_classification_max = max(crack_classification.keys(), key=(lambda k: crack_classification[k]))

            # Make 2D crack map
            if classification_max == 'patch':
                crack_map[y][x] = CLASS_LIST.index(classification_max)
            elif classification_max == 'crack':
                crack_map[y][x] = CLASS_LIST.index(crack_classification_max)

        # Start region proposing algorithm
        # Cack map scan
        crack_map = region_numbering(crack_map, axis_x, axis_y, connectivity_option)
        # Remove duplicate region
        crack_map = check_nearby_region(crack_map, axis_x, axis_y, connectivity_option)
        # Noise filter
        crack_map = noise_filtering(crack_map, axis_x, axis_y, noise_filtering_option)

        region_result = []

        # Put json data
        for image_info in image_infos:
            current_x = int(image_info['position']['x'] / image_info['position']['w'])
            current_y = int(image_info['position']['y'] / image_info['position']['h'])
            severity = {}
            if 'severity' in image_info:
                severity = image_info['severity']
            current_value = crack_map[current_y][current_x]

            # if current_value is crack or patch put to json
            if current_value > 0:
                is_json_region_not_exist = True

                # if region exist, add patch info to region area
                for i in range(0, len(region_result)):
                    if region_result[i]['region'] == current_value:
                        is_json_region_not_exist = False
                        region_result[i]['region_area'].append({
                                                                   'h': patch_size, 'w': patch_size,
                                                                   'x': current_x * patch_size,
                                                                   'y': current_y * patch_size,
                                                                   'severity': severity
                                                               } if (len(severity) > 0) else {
                            'h': patch_size, 'w': patch_size,
                            'x': current_x * patch_size, 'y': current_y * patch_size,
                        })

                # if region not exist, make region and append patch info
                if is_json_region_not_exist:
                    region_result.append({
                                             'region': current_value,
                                             'region_type': CLASS_LIST[int(current_value / 100)],
                                             'region_area': [{
                                                 'h': patch_size, 'w': patch_size,
                                                 'x': current_x * patch_size, 'y': current_y * patch_size,
                                                 'severity': severity
                                             }]
                                         } if (len(severity) > 0) else {
                        'region': current_value,
                        'region_type': CLASS_LIST[int(current_value / 100)],
                        'region_area': [{
                            'h': patch_size, 'w': patch_size,
                            'x': current_x * patch_size, 'y': current_y * patch_size,
                        }]})

        # region thresholding process

        region_result = region_thresholding_process(region_result, region_threshold)

        # region number sorting and severity processing
        for i in range(0, len(region_result)):
            region_result[i]['region'] = i
            region_type = region_result[i]['region_type']
            max_width_x = []
            max_width_y = []
            maxx = []
            maxy = []
            minx = []
            miny = []
            total_average_width = []
            total_max_width = []
            region_area_infos = region_result[i]['region_area']
            # print('p', len(region_area_infos))
            # if len(region_area_infos) < region_threshold :
            # del region_result[i]
            is_crack = False
            is_patch = False

            for region_area_info in region_area_infos:
                # if is crack
                if 'severity' in region_area_info and (
                        region_type == 'ac' or region_type == 'tc' or region_type == 'lc'):
                    is_crack = True
                    # print(i, region_area_info['severity'])
                    max_width_x.append(region_area_info['severity']['max_width_x'] + float(region_area_info['x']))
                    max_width_y.append(region_area_info['severity']['max_width_y'] + float(region_area_info['y']))
                    maxx.append(region_area_info['severity']['maxx'] + float(region_area_info['x']))
                    maxy.append(region_area_info['severity']['maxy'] + float(region_area_info['y']))
                    minx.append(region_area_info['severity']['minx'] + float(region_area_info['x']))
                    miny.append(region_area_info['severity']['miny'] + float(region_area_info['y']))
                    total_max_width.append(region_area_info['severity']['total_max_width'])
                    total_average_width.append(region_area_info['severity']['total_average_width'])

                # if is patch
                elif 'severity' not in region_area_info and region_type == 'patch':
                    is_patch = True
                    minx.append(region_area_info['x'])
                    maxx.append(region_area_info['x'] + region_area_info['w'])
                    miny.append(region_area_info['y'])
                    maxy.append(region_area_info['y'] + region_area_info['h'])
                else:
                    print('Region data error', 'region_type : ', region_type)

            if is_crack:
                # print(total_max_width)
                max_of_total_max_width = max(total_max_width)
                # if max_of_total_max_width != 0:
                region_result[i]['total_max_width'] = max_of_total_max_width
                region_result[i]['max_width_x'] = max_width_x[total_max_width.index(max_of_total_max_width)]
                region_result[i]['max_width_y'] = max_width_y[total_max_width.index(max_of_total_max_width)]

                region_result[i]['total_average_width'] = sum(total_average_width) / len(total_average_width)
                region_result[i]['maxx'] = max(maxx)
                region_result[i]['maxy'] = max(maxy)
                region_result[i]['minx'] = min(minx)
                region_result[i]['miny'] = min(miny)
                region_result[i]['area'] = (max(maxx) - min(minx)) * (max(maxy) - min(miny))
                region_result[i]['severity'] = crack_region_severity(max_of_total_max_width)

            # if is patch
            elif is_patch:
                patching_region = [min(minx), min(miny), max(maxx), max(maxy)]
                area, bbox, contour, patching_seg_image = patching_region_severity(input_image, patching_region,
                                                                                    patch_size)
                # print((contour))
                patching_bbox_minx, patching_bbox_miny, patching_bbox_maxx, patching_bbox_maxy = bbox
                region_result[i]['area'] = area
                region_result[i]['patching_region_minx'] = int(patching_region[0])
                region_result[i]['patching_region_miny'] = int(patching_region[1])
                region_result[i]['patching_region_maxx'] = int(patching_region[2])
                region_result[i]['patching_region_maxy'] = int(patching_region[3])

                region_result[i]['patching_bbox_minx'] = int(patching_bbox_minx + patching_region[0])
                region_result[i]['patching_bbox_miny'] = int(patching_bbox_miny + patching_region[1])
                region_result[i]['patching_bbox_maxx'] = int(patching_bbox_maxx + patching_region[0])
                region_result[i]['patching_bbox_maxy'] = int(patching_bbox_maxy + patching_region[1])
                # for c in contour:
                #     c[0] = patching_region[0] + c[0]
                #     c[1] = patching_region[1] + c[1]
                # region_result[i]['contours'] = contour
                # print(contour)
                region_result[i]['patching_seg_image'] = str(patching_seg_image)

        return region_result


def coordinate_checker(current_x, current_y, max_x, max_y):
    if (current_x >= 0) and (current_x < max_x) and (current_y >= 0) and (current_y < max_y):
        return True
    return False


def region_numbering(crack_map, axis_x, axis_y, connectivity_option):
    num_of_different_region = [0 for x in range(CLASS_NUM)]
    for current_x in range(0, axis_x):
        for current_y in range(0, axis_y):
            is_connected_component_exist = False
            current_value = crack_map[current_y][current_x]

            # keep scan if current_value is not (normal or lane)
            if current_value > 0:
                # 4 or 8 connectibity scan
                for check_pos_add_to_x in CHECK_POSITION_LIST:
                    for check_pos_add_to_y in CHECK_POSITION_LIST:

                        if (
                                connectivity_option == OPTION_CONNECTIVITY_4) and check_pos_add_to_x * check_pos_add_to_y != 0:
                            continue
                        check_pos_x = current_x + check_pos_add_to_x
                        check_pos_y = current_y + check_pos_add_to_y

                        # index check, an index must inside crack map
                        if coordinate_checker(check_pos_x, check_pos_y, axis_x, axis_y):
                            check_pos_value = crack_map[check_pos_y][check_pos_x]

                            # if checking pos has region name, than change curren pos region name
                            if int(check_pos_value / 100) == current_value:
                                crack_map[current_y][current_x] = check_pos_value
                                is_connected_component_exist = True

                # if there is no region name make new one
                if is_connected_component_exist == False:
                    crack_map[current_y][current_x] = current_value * 100 + num_of_different_region[current_value]
                    num_of_different_region[current_value] += 1
    return crack_map


def check_nearby_region(crack_map, axis_x, axis_y, connectivity_option):
    for current_x in range(0, axis_x):
        for current_y in range(0, axis_y):
            current_value = crack_map[current_y][current_x]

            if current_value > 0:
                for check_pos_add_to_x in CHECK_POSITION_LIST:
                    for check_pos_add_to_y in CHECK_POSITION_LIST:
                        if (
                                connectivity_option == OPTION_CONNECTIVITY_4) and check_pos_add_to_x * check_pos_add_to_y != 0:
                            continue

                        check_pos_x = current_x + check_pos_add_to_x
                        check_pos_y = current_y + check_pos_add_to_y

                        if coordinate_checker(check_pos_x, check_pos_y, axis_x, axis_y):
                            check_pos_value = crack_map[check_pos_y][check_pos_x]
                            if int(check_pos_value / 100) == int(current_value / 100) and current_value > 10:
                                for change_x in range(0, axis_x):
                                    for change_y in range(0, axis_y):
                                        if crack_map[change_y][change_x] == current_value:
                                            crack_map[change_y][change_x] = check_pos_value
    return crack_map


# TODO : Fix some error, debug through web site
def noise_filtering(crack_map, axis_x, axis_y, noise_filtering_option):
    if noise_filtering_option != OPTION_NOISE_FILTER_NOTHING:
        for current_x in range(0, axis_x):
            for current_y in range(0, axis_y):
                zero_count = 0
                region_count = {}
                current_value = crack_map[current_y][current_x]

                if (current_value > 0) and ((noise_filtering_option == OPTION_NOISE_FILTER_CUT_EDGE) or (
                        noise_filtering_option == OPTION_NOISE_FILTER_CUT_EDGE_AND_FILL_IN_THE_BLANK)):
                    for check_pos_add_to_x in CHECK_POSITION_LIST:
                        for check_pos_add_to_y in CHECK_POSITION_LIST:
                            if check_pos_add_to_x * check_pos_add_to_y != 0:
                                continue
                            check_pos_x = current_x + check_pos_add_to_x
                            check_pos_y = current_y + check_pos_add_to_y

                            if coordinate_checker(check_pos_x, check_pos_y, axis_x, axis_y):
                                check_pos_value = crack_map[check_pos_y][check_pos_x]
                                if check_pos_value == 0:
                                    zero_count += 1

                    if zero_count >= 3:
                        crack_map[current_y][current_x] = 0
                    zero_count = 0
                elif (current_value == 0) and ((noise_filtering_option == OPTION_NOISE_FILTER_FILL_IN_THE_BLANK) or (
                        noise_filtering_option == OPTION_NOISE_FILTER_CUT_EDGE_AND_FILL_IN_THE_BLANK)):
                    for check_pos_add_to_x in CHECK_POSITION_LIST:
                        for check_pos_add_to_y in CHECK_POSITION_LIST:
                            if check_pos_add_to_x * check_pos_add_to_y != 0:
                                continue
                            check_pos_x = current_x + check_pos_add_to_x
                            check_pos_y = current_y + check_pos_add_to_y

                            if coordinate_checker(check_pos_x, check_pos_y, axis_x, axis_y):
                                check_pos_value = crack_map[check_pos_y][check_pos_x]
                                if check_pos_value > 0:
                                    if str(check_pos_value) in region_count:
                                        region_count[str(check_pos_value)] += 1
                                    else:
                                        region_count[str(check_pos_value)] = 1

                    if len(region_count) > 0:
                        v = list(region_count.values())
                        if max(v) >= 3:
                            # print("?")
                            k = list(region_count.keys())
                            crack_map[current_y][current_x] = int(k[v.index(max(v))])
                    region_count = {}
    return crack_map


def crack_region_severity(max_of_total_max_width):
    if max_of_total_max_width <= 6:
        severity = 'low'
    elif max_of_total_max_width > 6 and max_of_total_max_width <= 19:
        severity = 'medium'
    else:
        severity = 'high'
    return severity


def patching_region_severity(input_image, patching_region, patch_size):
    area = (patching_region[0], patching_region[1], patching_region[2], patching_region[3])
    cropped_image = input_image.crop(area)
    np_image = np.array(cropped_image)
    area = 0
    bbox = [0, 0, 0, 0]
    contours = [[[0, 1], [0, 1]]]
    seg_image = 0
    contour = []

    # if image exist, do work
    if len(np_image) > 0 and len(np_image.shape) < 3:
        dest_height, dest_width = np_image.shape

        ret, threshed_img = cv2.threshold(np_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # make filter size dynamic correspond to patch_size
        gaussion_filter_size = 65
        if (patch_size / 4) % 2 == 0:
            gaussion_filter_size = int((patch_size / 4) + 1)
        else:
            gaussion_filter_size = int(patch_size / 4)
        max_iter = 20

        for i in range(0, max_iter):
            threshed_img = cv2.GaussianBlur(threshed_img, (gaussion_filter_size, gaussion_filter_size), 0)
            ret3, threshed_img = cv2.threshold(threshed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # cv2.imwrite(str(patching_region[0]) + str(patching_region[1]) + str(patching_region[2]) + '_filter_size_'+ str(gaussion_filter_size) + '_iter_' + str(i) +'_.jpg', threshed_img)
            # print(i, (threshed_img == 0).sum())
            # print(i, (threshed_img == 255).sum())
        # cropped_image.show()
        # cv2.imshow("contours", np_image)

        # To make bbox more pretty
        threshed_img_boder = cv2.copyMakeBorder(threshed_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        height, width = threshed_img.shape
        threshed_img_boder_reshape = cv2.resize(threshed_img_boder, None, fx=dest_width / width,
                                                fy=dest_height / height, interpolation=cv2.INTER_AREA)
        seg_image = base64.b64encode(threshed_img_boder_reshape)

        contours, hier = cv2.findContours(threshed_img_boder_reshape, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        len_of_contours = []
        # TODO : not to use x, y
        x = []
        y = []
        for c in contours:
            len_of_contours.append(len(c))

        # if contours is exist, do work
        if len(len_of_contours) > 0:
            contours = [contours[len_of_contours.index(max(len_of_contours))]]
            for c in contours[0]:
                x.append(c[0][0])
                y.append(c[0][1])
                contour.append([int(c[0][0]), int(c[0][1])])

            # TODO : Use this function, if you want to draw a shape on the frontend view
            # cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

            bbox = min(x), min(y), max(x), max(y)
            area = cv2.contourArea(contours[0])
        else:
            print("Patching segmentation error : contours not exist!!")
    elif len(np_image.shape) == 3:
        print("Patching image cropping error : input gray image")
    else:
        print('Patching image cropping error : cropped_image not exist!!')

    return area, bbox, contour, seg_image


def region_thresholding_process(region_result, region_threshold):
    j = 0
    while (j < len(region_result)):
        # print('j', j)
        if len(region_result[j]['region_area']) < region_threshold:
            del region_result[j]
            j -= 1
        j += 1
    return region_result