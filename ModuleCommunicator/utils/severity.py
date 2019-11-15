from cv2 import cv2
import numpy as np
from PIL import Image
import math
import glob
import os
import time
import csv
import base64
from io import BytesIO

def rotate_image_and_crop(mat, angle, bound_w,bound_h):
    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (width / 2, height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h),borderMode=cv2.BORDER_CONSTANT)

    return rotated_mat

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h),flags=cv2.INTER_NEAREST)

    return rotated_mat


def get_max_average_width_of_crack(display,number_of_points):
    """
    ## INPUT : patch_img_path : 'display : height,width shape (numpy array) , number_of_iterations : 50 (int)
    ## OUTPUT : MAX WIDTH , AVERAGE WIDTH , MIN X, MIN Y, MAX X, MAX Y, MAX WIDTH X, MAX WIDTH Y
    """

    display_size = display.shape

    index= np.nonzero(display>1)
    points = np.zeros((len(index[0]), 2), dtype=np.float32)
    index_y = index[0]
    index_x = index[1]
    if len(index_x)==0:
        return 0,0,0,0,0,0,0,0
        '''return total_max_width,total_average_width,minx,miny,maxx,maxy,max_width_x,max_width_y'''
    minx = min(index_x)
    maxx = max(index_x)
    miny = min(index_y)
    maxy = max(index_y)

    for i in range(0,len(index[0])):
        y=index[0][i]
        x=index[1][i]
        points[i, :] = [x, y]

    # cv2.imshow('Display', display)
    # cv2.waitKey(0)

    # print(points.shape)
    # minimum inlier distance and iterations
    eeta = 20
    iterations = int(number_of_points / 2)

    # Initializing best params
    max_inliers = 0
    best_m = 0
    best_b = 0

    # Iterations begin
    for i in range(iterations):
        # Selecting random samples (two)
        r1 = np.random.randint(0, points.shape[0])
        r2 = np.random.randint(0, points.shape[0] - 1)
        if r2 == r1:
            r2 += 1

        point_1 = points[r1, :]
        point_2 = points[r2, :]

        # Calculating new values of m and b from random samples
        if (point_2[0] - point_1[0])!=0:
            m = (point_2[1] - point_1[1]) / (point_2[0] - point_1[0])
            b = -m * point_1[0] + point_1[1]

            # Finding difference (perpendicular distance of point to line)
            diff = abs(((-m * points[:, 0]) + (points[:, 1] - b)) / ((-m) ** 2 + 1) ** 0.5)

            # Calculating inliers
            inliers = len(np.where(diff < eeta)[0])

            # Updating best params if better inliers found
            if inliers > max_inliers:
                max_inliers = inliers
                best_m = m
                best_b = b

    # display.flags.writeable = True
    p1 = (0, int(best_b))
    p2 = (display_size[0], int(best_m * display_size[0] + best_b))
    cv2.line(display, p1, p2, (0, 0, 255), 2)
    ###############################################################
    #  Find Tangent's Degree
    ###############################################################

    radian = np.arctan(best_m)
    degree = radian * (180 / math.pi )
    # print('origin size : ',display_size)
    # print('radian : {}'.format(radian))
    # print('degree : {}'.format(degree))

    ###############################################################
    # Create a new boundary and rotate the image so that the crack is horizontal.
    ###############################################################

    display2=rotate_image(display,degree)
    display2[display2!=0]=255
    # print(np.unique(display,return_counts=True))

    display2_size = display2.shape
    # print('rotated size : ',display_size)

    total_max_width, total_average_width = 0, 0

    ###############################################################
    # Find the maximum width and average width of the crack.
    ###############################################################
    max_widths =[]
    num_of_crack_pixel = 0
    max_width_x ,max_width_y = 0,0
    index = np.nonzero(display2 > 1)
    if len(index[0])>0:
        index_y = index[0]
        index_x = index[1]

        minx = min(index_x)
        maxx = max(index_x)
        miny = min(index_y)
        maxy = max(index_y)

        for j in range(minx-1,maxx+1):
            max_width = 0
            tmp_width = 0
            isCrack = False
            for i in range(miny-1,maxy+1):
                value = display2[i, j]
                if value==255:
                    isCrack = True
                    num_of_crack_pixel+=1
                    tmp_width+=1
                    max_width = tmp_width
                else :
                    tmp_width = 0
            if isCrack:
                max_widths.append(max_width)
            if(total_max_width<max_width):
                total_max_width=max_width
                tmp_width = 0
                for i in range(miny-1,maxy+1):
                    value = display2[i, j]
                    if value == 255:
                        isCrack = True
                        tmp_width +=1
                        if tmp_width==total_max_width/2:
                            display2[max_width_y,max_width_x]=255
                            max_width_x = j
                            max_width_y = i
                            display2[max_width_y,max_width_x]=128

                else:
                    tmp_width=0

    # cv2.rectangle(display2, (max_width_x - 1, max_width_y - 1), (max_width_x + 1, max_width_y + 1), (0, 0, 255), 1)
    display2[display2==255]=0
    display2[display2!=0] = 255

    display3=rotate_image_and_crop(display2,-degree,display_size[0],display_size[1])
    max_width_y,max_width_x=np.where(display3==np.unique(display3)[-1])
    max_width_x, max_width_y = max_width_x[0], max_width_y[0]


    if num_of_crack_pixel>0:
        total_average_width = sum(max_widths) / len(max_widths)
    else:
        pass
        # print('No crack pixels.')

    # cv2.rectangle(display, (max_width_x - 1, max_width_y - 1), (max_width_x + 1, max_width_y + 1), (0, 0, 255), 1)

    return total_max_width,total_average_width,minx,miny,maxx,maxy,max_width_x,max_width_y

def crack_width_analysis(seg_image, threshold, cls_result_data, iter=30,patch_size=256):
    image = base64.b64decode(seg_image)

    full_img_dict = []
    input_img = Image.open(BytesIO(image)).convert('L')
    display = np.asarray(input_img)
    display.flags.writeable = True
    display[display <= threshold] = 0
    cv2.imwrite("test_" + str(threshold) + ".png", display)

    width, height = input_img.size
    patch_width = int(width / patch_size)
    patch_height = int(height / patch_size)
    count = 0

    for j in range(0, patch_height):
        for i in range(0, patch_width):
            y = patch_size * j
            x = patch_size * i

            for k in range(len(cls_result_data)) :
                cls_position = cls_result_data[k]['position']
                crack = cls_result_data[k]
                labels = sorted(crack['label'], key=lambda label_list: (label_list['score']), reverse=True)
                if labels[0]['description'] == 'crack' :
                    if cls_position['x'] == x and cls_position['y'] == y:
                        patch_img = display[patch_size * j:patch_size * (j + 1), patch_size * i:patch_size * (i + 1)]
                        total_max_width, total_average_width, minx, miny, maxx, maxy, max_width_x, max_width_y = get_max_average_width_of_crack(patch_img, iter)
                        full_img_dict.append({
                            'x':patch_size*i,
                            'y':patch_size*j,
                            'w':patch_size,
                            'h':patch_size,
                            'total_max_width':total_max_width,
                            'total_average_width':total_average_width,
                            'minx':minx,
                            'miny':miny,
                            'maxx':maxx,
                            'maxy':maxy,
                            'max_width_x':max_width_x,
                            'max_width_y':max_width_y
                        })
                        count += 1
    return full_img_dict, count