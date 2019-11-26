import base64
from io import BytesIO
import cv2
import json
import random
import numpy
from PIL import Image 

def color():
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    rect_color = [r, g, b]
    text_color = [255 - r, 255 - g, 255 - b]

    return rect_color, text_color

def make_result_image(region_results, seg_image) :
    image = base64.b64decode(seg_image)
    pil_image = Image.open(BytesIO(image)).convert('RGB')
    open_cv_image = numpy.array(pil_image)
    seg_image = open_cv_image[:, :, ::-1].copy()
    for region_result in region_results:
        region_num = str(region_result['region'])
        region_type = region_result['region_type']
        patches = region_result['region_area']
        xs = []
        ys = []

        rect_color, text_color = color()

        for patch in patches:
            x = patch['x']
            y = patch['y']
            w = patch['w']
            h = patch['h']
            xs.append(x)
            ys.append(y)
            img = cv2.rectangle(seg_image, (x, y), (x + w, y + h), rect_color, 10)

        min_x = min(xs)
        min_y = min(ys)
        max_x = max(xs)
        max_y = max(ys)

        center_x = int(min_x + (max_x - min_x) / 2) - 32
        center_y = int(min_y + (max_y - min_y) / 2) + 224

        if region_result['region_type'] != 'patch':
            severity = region_result['severity']
            severity_result = ''
            if severity == 'low':
                severity_result = 'L'
            elif severity == 'medium':
                severity_result = 'M'
            elif severity == 'high':
                severity_result = 'H'
            cv2.putText(seg_image, region_num + "_" + region_type.upper() + "(" + severity_result + ")",
                        (center_x, center_y), cv2.FONT_HERSHEY_DUPLEX, 7, text_color, 7)
        else:
            cv2.putText(seg_image, region_num + "_" + region_type.upper(), (center_x, center_y),
                        cv2.FONT_HERSHEY_DUPLEX, 7, text_color, 7)
    cv2_im = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)

    buffered = BytesIO()
    pil_im.save(buffered, format="PNG")
    result_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return result_image