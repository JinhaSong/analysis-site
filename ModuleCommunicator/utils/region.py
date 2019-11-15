import json
import itertools

def make_region(cls_result_data, connectivity_option=1, noise_filter_option=3):
    CLASS_LIST = ['normal', 'crack', 'patch', 'lane', 'detail_norm', 'ac', 'tc', 'lc']
    CLASS_NUM = len(CLASS_LIST)
    # TODO: check below information
    # input json is from mltwins:8222/ModuleCommunicator/task.py/def:communicator/line:27 - cls_result_data

    OPTION_CONNECTIVITY_4 = 0
    OPTION_CONNECTIVITY_8 = 1

    OPTION_NOISE_FILTER_NOTHING = 0
    OPTION_NOISE_FILTER_FILL_IN_THE_BLANK = 1
    OPTION_NOISE_FILTER_CUT_EDGE = 2
    OPTION_NOISE_FILTER_CUT_EDGE_AND_FILL_IN_THE_BLANK = 3

    image_infos = cls_result_data['results'][0]['module_result']

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
    # print('----------------------------------------------------------')
    # print out the result data
    # for c in crack_map:
    #     print(c)

    check_pos_list = [-1, 0, 1]
    num_of_different_region = [0 for x in range(CLASS_NUM)]

    # region proposing algorithm

    # crack map scan
    for current_x in range(0, axis_x):
        for current_y in range(0, axis_y):
            is_connected_component_exist = False
            current_value = crack_map[current_y][current_x]

            # keep scan if current_value is not (normal or lane)
            if current_value > 0:
                # 4 or 8 connectibity scan
                for check_pos_add_to_x in check_pos_list:
                    for check_pos_add_to_y in check_pos_list:

                        if (
                                connectivity_option == OPTION_CONNECTIVITY_4) and check_pos_add_to_x * check_pos_add_to_y != 0:
                            continue
                        check_pos_x = current_x + check_pos_add_to_x
                        check_pos_y = current_y + check_pos_add_to_y
                        # index check, an index must inside crack map
                        if check_pos_x >= 0 and check_pos_x < axis_x and check_pos_y >= 0 and check_pos_y < axis_y:
                            check_pos_value = crack_map[check_pos_y][check_pos_x]
                            # if checking pos has region name, than change curren pos region name
                            if int(check_pos_value / 100) == current_value:
                                crack_map[current_y][current_x] = check_pos_value
                                is_connected_component_exist = True
                # if there is no region name make new one
                if is_connected_component_exist == False:
                    crack_map[current_y][current_x] = current_value * 100 + num_of_different_region[current_value]
                    num_of_different_region[current_value] += 1

    # remove duplicate region
    result = []
    # print(len(result))
    for current_x in range(0, axis_x):
        for current_y in range(0, axis_y):
            current_value = crack_map[current_y][current_x]

            if current_value > 0:
                for check_pos_add_to_x in check_pos_list:
                    for check_pos_add_to_y in check_pos_list:
                        if (
                                connectivity_option == OPTION_CONNECTIVITY_4) and check_pos_add_to_x * check_pos_add_to_y != 0:
                            continue

                        check_pos_x = current_x + check_pos_add_to_x
                        check_pos_y = current_y + check_pos_add_to_y

                        if check_pos_x >= 0 and check_pos_x < axis_x and check_pos_y >= 0 and check_pos_y < axis_y:
                            check_pos_value = crack_map[check_pos_y][check_pos_x]
                            if int(check_pos_value / 100) == int(current_value / 100) and current_value > 10:
                                for change_x in range(0, axis_x):
                                    for change_y in range(0, axis_y):
                                        if crack_map[change_y][change_x] == current_value:
                                            crack_map[change_y][change_x] = check_pos_value

    # noise filter
    if noise_filter_option != OPTION_NOISE_FILTER_NOTHING:
        for current_x in range(0, axis_x):
            for current_y in range(0, axis_y):
                zero_count = 0
                region_count = {}
                current_value = crack_map[current_y][current_x]

                if (current_value > 0) and ((noise_filter_option == OPTION_NOISE_FILTER_CUT_EDGE) or (
                        noise_filter_option == OPTION_NOISE_FILTER_CUT_EDGE_AND_FILL_IN_THE_BLANK)):
                    for check_pos_add_to_x in check_pos_list:
                        for check_pos_add_to_y in check_pos_list:

                            if check_pos_add_to_x * check_pos_add_to_y != 0:
                                continue

                            check_pos_x = current_x + check_pos_add_to_x
                            check_pos_y = current_y + check_pos_add_to_y

                            if check_pos_x >= 0 and check_pos_x < axis_x and check_pos_y >= 0 and check_pos_y < axis_y:
                                check_pos_value = crack_map[check_pos_y][check_pos_x]
                                if check_pos_value == 0:
                                    zero_count += 1

                    if zero_count >= 3:
                        crack_map[current_y][current_x] = 0
                    zero_count = 0
                elif (current_value == 0) and ((noise_filter_option == OPTION_NOISE_FILTER_FILL_IN_THE_BLANK) or (
                        noise_filter_option == OPTION_NOISE_FILTER_CUT_EDGE_AND_FILL_IN_THE_BLANK)):
                    for check_pos_add_to_x in check_pos_list:
                        for check_pos_add_to_y in check_pos_list:

                            if check_pos_add_to_x * check_pos_add_to_y != 0:
                                continue

                            check_pos_x = current_x + check_pos_add_to_x
                            check_pos_y = current_y + check_pos_add_to_y

                            if check_pos_x >= 0 and check_pos_x < axis_x and check_pos_y >= 0 and check_pos_y < axis_y:
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

    # make result json
    for image_info in image_infos:
        current_x = int(image_info['position']['x'] / image_info['position']['w'])
        current_y = int(image_info['position']['y'] / image_info['position']['h'])
        # label = image_info['label']
        # for current_x in range(0, axis_x):
        #     for current_y in range(0, axis_y):
        current_value = crack_map[current_y][current_x]
        if current_value > 0:
            is_json_region_not_exist = True
            if len(result) > 0:
                for i in range(0, len(result)):
                    if result[i]['region'] == current_value:
                        is_json_region_not_exist = False
                        result[i]['region_area'].append({
                            'h': patch_size, 'w': patch_size,
                            'x': current_x * patch_size, 'y': current_y * patch_size,
                            # 'label': label
                        })

            if len(result) == 0 or is_json_region_not_exist:
                result.append({
                    'region': current_value,
                    'region_type': CLASS_LIST[int(current_value / 100)],
                    'region_area': [{
                        'h': patch_size, 'w': patch_size,
                        'x': current_x * patch_size, 'y': current_y * patch_size,
                        # 'label': label
                    }]
                })

    for i in range(0, len(result)):
        result[i]['region'] = i

    ex_list = list(itertools.chain.from_iterable(crack_map))
    ex_list = list(set(ex_list))

    # print(ex_list)
    # print(len(ex_list))
    # print('----------------------------------------------------------')
    # print out the result data
    for c in crack_map:
        for i in range(0, len(c)):
            if c[i] == 0:
                c[i] = '000'
            else:
                c[i] = str(c[i])

    return result
