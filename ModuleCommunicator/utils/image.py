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

def make_result_image(region_results, severity_threshold, str_seg_image) :
    image = base64.b64decode(str_seg_image)
    input_img = Image.open(BytesIO(image)).convert('L')
    display = numpy.asarray(input_img)
    display.flags.writeable = True
    display[display < severity_threshold] = 0
    
    pil_image = Image.fromarray(display).convert('RGB')

    open_cv_image = numpy.array(pil_image)
    seg_image = open_cv_image[:, :, ::-1].copy()
    
    for region_result in region_results:
        region_num = str(region_result['region'])
        region_type = region_result['region_type'][:3]
        patches = region_result['region_area']
        ys = []

        rect_color = None
        if region_type == "ac" :
            rect_color = [0,0,255]
        elif region_type == "lc" :
            rect_color = [0,255,255]
        elif region_type == "tc" :
            rect_color = [71,173,112]
        elif region_type == "patch":
            rect_color = [255,255,255]
        elif region_type == "pot" :
            rect_color = [240,176,0]
        
        min_x = 0
        min_y = 0
        count = 0
        for patch in patches:
            x = patch['x']
            y = patch['y']
            w = patch['w']
            h = patch['h']
            
            if count == 0 :
                min_x = x
                min_y = y + 100
            count+=1

            img = cv2.rectangle(seg_image, (x, y), (x + w, y + h), rect_color, 10)


        if region_result['region_type'] != 'patch':
            severity = region_result['severity']
            severity_result = ''
            if severity == 'low':
                severity_result = 'L'
            elif severity == 'medium':
                severity_result = 'M'
            elif severity == 'high':
                severity_result = 'H'
            cv2.putText(seg_image, region_type.upper() + "(" + severity_result + ")",
                        (min_x, min_y), cv2.FONT_HERSHEY_DUPLEX, 2, rect_color, 2)
        else:
            cv2.putText(seg_image, region_type.upper(), (min_x, min_y),
                        cv2.FONT_HERSHEY_DUPLEX, 2, rect_color, 2)
    cv2_im = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)

    buffered = BytesIO()
    pil_im.save(buffered, format="PNG")
    result_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return result_image