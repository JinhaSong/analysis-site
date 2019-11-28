import base64
from io import BytesIO
import cv2
import json
import random
import numpy
from PIL import Image, ImageOps

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
        region_type = region_result['region_type']
        patches = region_result['region_area']

        rect_color = None
        if region_type == "ac" :
            rect_color = [0,0,255]
        elif region_type == "lc" :
            rect_color = [0,255,255]
        elif region_type == "tc" :
            rect_color = [71,173,112]
        elif region_type == "patch":
            rect_color = [255,0,0]
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
                min_x = x + 50
                min_y = y + 100
            count+=1

            img = cv2.rectangle(seg_image, (x, y), (x + w, y + h), rect_color, 10)

        region_type = region_type[:3]
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
    tmp = cv2.cvtColor(seg_image, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 200, cv2.THRESH_BINARY)
    b, g, r = cv2.split(seg_image)
    rgba = [b, g, r, alpha]
    cv2_im = cv2.merge(rgba, 4)
    pil_im = Image.fromarray(cv2_im)

    buffered = BytesIO()
    pil_im.save(buffered, format="PNG")
    result_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return result_image

def attach_patch(region_results, severity_threshold, str_seg_image) :
    patches = []
    for region_result in region_results :
        if region_result['region_type'] == "patch" :
            patches.append(region_result)

    str_image = base64.b64decode(str_seg_image)
    image = Image.open(BytesIO(str_image)).convert('RGB')
    background = numpy.array(image)
    background_copy = background[:, :, ::-1].copy()

    result = None
    result_image = ''
    for i in range(len(patches)):
        str_image = base64.b64decode(patches[i]['patching_seg_image'])
        image = Image.open(BytesIO(str_image)).convert('RGB')
        patch = numpy.array(image)
        patch = patch[:, :, ::-1].copy()
    
        x = int(patches[i]['patching_region_minx'])
        y = int(patches[i]['patching_region_miny'])
        background_gray = cv2.cvtColor(background_copy, cv2.COLOR_BGR2GRAY)

        patch_inverse = numpy.where(patch == 255, 0, 255).astype('uint8')  # Invert patch
        patch_inverse = cv2.cvtColor(patch_inverse, cv2.COLOR_BGR2GRAY)  # patch grayscale transform

        result = image_blending(background_gray, patch_inverse, x, y).copy()

    pil_im = Image.fromarray(result.copy())

    buffered = BytesIO()
    pil_im.save(buffered, format="PNG")
    result_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return result_image


def image_blending(background, patch, x, y):
   _, mask_inv = cv2.threshold(patch, 10, 255, cv2.THRESH_BINARY_INV)

   patch_height, patch_width = patch.shape
   roi = background[y: y + patch_height, x: x + patch_width]

   roi_patch = cv2.add(patch, roi, mask=mask_inv)
   result = cv2.add(roi_patch, patch)
   numpy.copyto(roi, result)

   return background

def convert_image_binary(str_seg_image) :
    image = base64.b64decode(str_seg_image)
    input_img = Image.open(BytesIO(image)).convert('RGB')
    display = numpy.asarray(input_img)
    display.flags.writeable = True
    display[display <= 127] = 0
    display[display > 127] = 255
    tmp = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 127, cv2.THRESH_BINARY)
    b, g, r = cv2.split(display)
    rgba = [b, g, r, alpha]
    seg_img = cv2.merge(rgba, 4)

    pil_im = Image.fromarray(seg_img)

    buffered = BytesIO()
    pil_im.save(buffered, format="PNG")
    result_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return result_image