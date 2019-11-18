# from cv2 import cv2
# import numpy as np
# from PIL import Image
# import math
# import glob
# import os
# import time
# import csv
# import base64
# from io import BytesIO
#
# def rotate_image_and_crop(mat, angle, bound_w,bound_h):
#     height, width = mat.shape[:2]  # image shape has 3 dimensions
#     image_center = (width / 2, height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
#
#     rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
#
#     rotation_mat[0, 2] += bound_w/2 - image_center[0]
#     rotation_mat[1, 2] += bound_h/2 - image_center[1]
#     rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h),borderMode=cv2.BORDER_CONSTANT)
#
#     return rotated_mat
#
# def rotate_image(mat, angle):
#     """
#     Rotates an image (angle in degrees) and expands image to avoid cropping
#     """
#
#     height, width = mat.shape[:2] # image shape has 3 dimensions
#     image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
#
#     rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
#
#     # rotation calculates the cos and sin, taking absolutes of those.
#     abs_cos = abs(rotation_mat[0,0])
#     abs_sin = abs(rotation_mat[0,1])
#
#     # find the new width and height bounds
#     bound_w = int(height * abs_sin + width * abs_cos)
#     bound_h = int(height * abs_cos + width * abs_sin)
#
#     # subtract old image center (bringing image back to origo) and adding the new image center coordinates
#     rotation_mat[0, 2] += bound_w/2 - image_center[0]
#     rotation_mat[1, 2] += bound_h/2 - image_center[1]
#
#     # rotate image with the new bounds and translated rotation matrix
#     rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h),flags=cv2.INTER_NEAREST)
#
#     return rotated_mat
#
#
# def get_max_average_width_of_crack(display,number_of_points):
#     """
#     ## INPUT : patch_img_path : 'display : height,width shape (numpy array) , number_of_iterations : 50 (int)
#     ## OUTPUT : MAX WIDTH , AVERAGE WIDTH , MIN X, MIN Y, MAX X, MAX Y, MAX WIDTH X, MAX WIDTH Y
#     """
#
#     display_size = display.shape
#
#     index= np.nonzero(display>1)
#     points = np.zeros((len(index[0]), 2), dtype=np.float32)
#     index_y = index[0]
#     index_x = index[1]
#     if len(index_x)==0:
#         return 0,0,0,0,0,0,0,0
#         '''return total_max_width,total_average_width,minx,miny,maxx,maxy,max_width_x,max_width_y'''
#     minx = min(index_x)
#     maxx = max(index_x)
#     miny = min(index_y)
#     maxy = max(index_y)
#
#     for i in range(0,len(index[0])):
#         y=index[0][i]
#         x=index[1][i]
#         points[i, :] = [x, y]
#
#     # cv2.imshow('Display', display)
#     # cv2.waitKey(0)
#
#     # print(points.shape)
#     # minimum inlier distance and iterations
#     eeta = 20
#     iterations = int(number_of_points / 2)
#
#     # Initializing best params
#     max_inliers = 0
#     best_m = 0
#     best_b = 0
#
#     # Iterations begin
#     for i in range(iterations):
#         # Selecting random samples (two)
#         r1 = np.random.randint(0, points.shape[0])
#         r2 = np.random.randint(0, points.shape[0] - 1)
#         if r2 == r1:
#             r2 += 1
#
#         point_1 = points[r1, :]
#         point_2 = points[r2, :]
#
#         # Calculating new values of m and b from random samples
#         if (point_2[0] - point_1[0])!=0:
#             m = (point_2[1] - point_1[1]) / (point_2[0] - point_1[0])
#             b = -m * point_1[0] + point_1[1]
#
#             # Finding difference (perpendicular distance of point to line)
#             diff = abs(((-m * points[:, 0]) + (points[:, 1] - b)) / ((-m) ** 2 + 1) ** 0.5)
#
#             # Calculating inliers
#             inliers = len(np.where(diff < eeta)[0])
#
#             # Updating best params if better inliers found
#             if inliers > max_inliers:
#                 max_inliers = inliers
#                 best_m = m
#                 best_b = b
#
#     # display.flags.writeable = True
#     p1 = (0, int(best_b))
#     p2 = (display_size[0], int(best_m * display_size[0] + best_b))
#     cv2.line(display, p1, p2, (0, 0, 255), 2)
#     ###############################################################
#     #  Find Tangent's Degree
#     ###############################################################
#
#     radian = np.arctan(best_m)
#     degree = radian * (180 / math.pi )
#     # print('origin size : ',display_size)
#     # print('radian : {}'.format(radian))
#     # print('degree : {}'.format(degree))
#
#     ###############################################################
#     # Create a new boundary and rotate the image so that the crack is horizontal.
#     ###############################################################
#
#     display2=rotate_image(display,degree)
#     display2[display2!=0]=255
#     # print(np.unique(display,return_counts=True))
#
#     display2_size = display2.shape
#     # print('rotated size : ',display_size)
#
#     total_max_width, total_average_width = 0, 0
#
#     ###############################################################
#     # Find the maximum width and average width of the crack.
#     ###############################################################
#     max_widths =[]
#     num_of_crack_pixel = 0
#     max_width_x ,max_width_y = 0,0
#     index = np.nonzero(display2 > 1)
#     if len(index[0])>0:
#         index_y = index[0]
#         index_x = index[1]
#
#         minx = min(index_x)
#         maxx = max(index_x)
#         miny = min(index_y)
#         maxy = max(index_y)
#
#         for j in range(minx-1,maxx+1):
#             max_width = 0
#             tmp_width = 0
#             isCrack = False
#             for i in range(miny-1,maxy+1):
#                 value = display2[i, j]
#                 if value==255:
#                     isCrack = True
#                     num_of_crack_pixel+=1
#                     tmp_width+=1
#                     max_width = tmp_width
#                 else :
#                     tmp_width = 0
#             if isCrack:
#                 max_widths.append(max_width)
#             if(total_max_width<max_width):
#                 total_max_width=max_width
#                 tmp_width = 0
#                 for i in range(miny-1,maxy+1):
#                     value = display2[i, j]
#                     if value == 255:
#                         isCrack = True
#                         tmp_width +=1
#                         if tmp_width==total_max_width/2:
#                             display2[max_width_y,max_width_x]=255
#                             max_width_x = j
#                             max_width_y = i
#                             display2[max_width_y,max_width_x]=128
#
#                 else:
#                     tmp_width=0
#
#     # cv2.rectangle(display2, (max_width_x - 1, max_width_y - 1), (max_width_x + 1, max_width_y + 1), (0, 0, 255), 1)
#     display2[display2==255]=0
#     display2[display2!=0] = 255
#
#     display3=rotate_image_and_crop(display2,-degree,display_size[0],display_size[1])
#     max_width_y,max_width_x=np.where(display3==np.unique(display3)[-1])
#     max_width_x, max_width_y = max_width_x[0], max_width_y[0]
#
#
#     if num_of_crack_pixel>0:
#         total_average_width = sum(max_widths) / len(max_widths)
#     else:
#         pass
#         # print('No crack pixels.')
#
#     # cv2.rectangle(display, (max_width_x - 1, max_width_y - 1), (max_width_x + 1, max_width_y + 1), (0, 0, 255), 1)
#
#     return total_max_width,total_average_width,minx,miny,maxx,maxy,max_width_x,max_width_y
#
# def crack_width_analysis(seg_image, threshold, cls_result_data, iter=30,patch_size=256):
#     image = base64.b64decode(seg_image)
#
#     full_img_dict = []
#     input_img = Image.open(BytesIO(image)).convert('L')
#     display = np.asarray(input_img)
#     display.flags.writeable = True
#     display[display <= threshold] = 0
#     cv2.imwrite("test_" + str(threshold) + ".png", display)
#
#     width, height = input_img.size
#     patch_width = int(width / patch_size)
#     patch_height = int(height / patch_size)
#     count = 0
#
#     for j in range(0, patch_height):
#         for i in range(0, patch_width):
#             y = patch_size * j
#             x = patch_size * i
#
#             for k in range(len(cls_result_data)) :
#                 cls_position = cls_result_data[k]['position']
#                 crack = cls_result_data[k]
#                 labels = sorted(crack['label'], key=lambda label_list: (label_list['score']), reverse=True)
#                 if labels[0]['description'] == 'crack' :
#                     if cls_position['x'] == x and cls_position['y'] == y:
#                         patch_img = display[patch_size * j:patch_size * (j + 1), patch_size * i:patch_size * (i + 1)]
#                         total_max_width, total_average_width, minx, miny, maxx, maxy, max_width_x, max_width_y = get_max_average_width_of_crack(patch_img, iter)
#                         full_img_dict.append({
#                             'x':patch_size*i,
#                             'y':patch_size*j,
#                             'w':patch_size,
#                             'h':patch_size,
#                             'total_max_width':total_max_width,
#                             'total_average_width':total_average_width,
#                             'minx':minx,
#                             'miny':miny,
#                             'maxx':maxx,
#                             'maxy':maxy,
#                             'max_width_x':max_width_x,
#                             'max_width_y':max_width_y
#                         })
#                         count += 1
#     return full_img_dict, count
import os
import cv2
import base64
import statistics
import numpy as np
from PIL import Image
from tqdm import tqdm
from io import BytesIO
import multiprocessing

def draw_rectangle(A,x,y):
    cv2.rectangle(A, (x - 1, y - 1), (x + 2, y + 2), (0, 0, 255), 1)
    return A

def get_circle_x_y_color_list(A,xCenter, yCenter, radius):
    xEdge = 0
    yEdge = radius
    p = 1 - radius
    height,width= A.shape
    circle_x_y_color =[]
    while(xEdge <= yEdge):
        xEdge = xEdge + 1
        if(p < 0):
            p = p + 2 * xEdge + 1
        else:
            yEdge = yEdge - 1
            p = p + 2 * (xEdge - yEdge) + 1

        xEdge = round(xEdge)
        yEdge = round(yEdge)

        x_y_dict = {'x_x_right':xCenter + xEdge,'y_y_down':yCenter + yEdge,'x_x_left':xCenter - xEdge,
                    'y_y_top': yCenter - yEdge,'y_x_down': yCenter + xEdge,'x_y_right' : xCenter + yEdge,
                    'y_x_top' : yCenter - xEdge, 'x_y_left': xCenter - yEdge}

        key = [['y_y_down','x_x_right'],['y_y_down', 'x_x_left'],['y_y_top','x_x_right'],
                    ['y_y_top', 'x_x_left'],['y_x_down', 'x_y_right'],['y_x_top', 'x_y_right'],
                    ['y_x_down', 'x_y_left'],['y_x_top', 'x_y_left']]
        white = (255, 255, 255)
        black = (0, 0, 0)
        color = None
        for idx,y_x in enumerate(key):
            y_key=y_x[0]
            x_key=y_x[1]
            y,x=round(x_y_dict[y_key]),round(x_y_dict[x_key])
            if x>0 and x<width and y>0 and y<height:
                current=A[y,x]
                if np.all(current==black):
                    color ='black'
                if np.all(current==white):
                    color = 'white'
                circle_x_y_color.append([x, y,color])

    *circle_x_y_color, = map(list, {*map(tuple, circle_x_y_color)})  # List de-duplication

    return circle_x_y_color

def get_is_joint(A,xCenter,yCenter,radius):
    circle_x_y_color=get_circle_x_y_color_list(A,xCenter,yCenter,radius)
    prev = None
    reverse_count = 0
    if not circle_x_y_color is None:
        median_x= int(statistics.median([x_y[0] for x_y in circle_x_y_color]))
        median_y = int(statistics.median([x_y[1] for x_y in circle_x_y_color]))
        one,two,three,four =[],[],[],[]

        for x_y_color in circle_x_y_color:
            x=x_y_color[0]
            y=x_y_color[1]
            color = x_y_color[2]
            if x >= median_x and y < median_y:  # 1사분면
                one.append([x,y,color])
            if x < median_x and y < median_y:  # 2사분면
                two.append([x, y, color])
            if x < median_x and y >= median_y:  # 3사분면
                three.append([x, y, color])
            if x >= median_x and y >= median_y:  # 4사분면
                four.append([x, y, color])

        one = sorted(one, key=lambda x: x[1], reverse=True)
        one = sorted(one, key=lambda x: x[0], reverse=True)
        two = sorted(two, key=lambda x: x[1])
        two = sorted(two, key=lambda x: x[0], reverse=True)
        three= sorted(three, key=lambda x: x[1])
        three = sorted(three, key=lambda x: x[0])
        four = sorted(four, key=lambda x: x[1], reverse=True)
        four = sorted(four, key=lambda x: x[0])
        new_list=one+two+three+four
        for i in new_list:
            if i[0]==0 or i[0]==255:
                prev = None
            elif i[1]==0 or i[1]==255:
                prev = None
            cur = i[2]
            if not prev is None and cur!=prev:
                # draw_rectangle(A,i[0],i[1])
                reverse_count+=1
            prev = cur
    if reverse_count > 4:
        is_joint = True
    else :
        is_joint = False
    return A,is_joint,reverse_count

def get_max_width_x_max_width_y(A,max_width_dict):
    if max_width_dict:
        values = sorted(max_width_dict.values(),reverse=True)
        for top_value in values:
            for key,value in max_width_dict.items():
                if value==top_value:
                    x_y=key.split('_')
                    x ,y = int(x_y[0]),int(x_y[1])
                    A,is_joint,reverse_count=get_is_joint(A,x,y,value)
                    if not is_joint :
                        return x,y,value

        for key, value in max_width_dict.items():
            if value == values[0]:
                x_y = key.split('_')
                x, y = int(x_y[0]), int(x_y[1])
                return x, y, value
    else:
        return 0,0,0
def count_length(arr,base_x,base_y,amount_of_change_x,amount_of_change_y,size_x,size_y):
    count = 0
    increase = 1
    pick_x, pick_y = base_x,base_y
    if abs(amount_of_change_x)==1 and abs(amount_of_change_y)==1:
        increase = 1.4142
    while arr[pick_y,pick_x] == 255 :
        pick_x +=amount_of_change_x
        pick_y +=amount_of_change_y
        if pick_y>=size_y or pick_x>=size_x :
            break
        if abs(pick_x - base_x)>35 or abs(pick_y - base_y)>35:
            break
        count+=increase
    return count

def every_search_get_max_avg_width_of_crack(A):
    """
    ## INPUT : patch_img_path : 'A'
    ## OUTPUT : MAX WIDTH , AVERAGE WIDTH , MIN X, MIN Y, MAX X, MAX Y, MAX WIDTH X, MAX WIDTH Y
    """
    max_width_dict = {}
    avg_crack_width,minx,miny,maxx,maxy =0,0,0,0,0
    A_size = A.shape
    A_height = A_size[0]
    A_width = A_size[1]
    index= np.nonzero(A>0)
    index_y = index[0]
    index_x = index[1]

    if len(index_x) == 0:  ## there are no cracks
        return A,avg_crack_width,minx,miny,maxx,maxy,max_width_dict
    minx, maxx, miny, maxy = min(index_x), max(index_x), min(index_y), max(index_y)
    num_of_crack_pixel = 0
    crack_width=[]

    for x,y in zip(index_x,index_y):
    # for x,y in zip([129,129,129,129,129,129,129,129,129,129,129,129],[184,186,188,190,192,194,196,198,200,202,204,206]):
        num_of_crack_pixel+=1
        top=count_length(A,x,y,0,-1,A_width,A_height)
        top_right = count_length(A,x,y,1,-1,A_width,A_height)
        right = count_length(A, x, y, 1, 0,A_width,A_height)
        right_down = count_length(A, x, y, 1, 1,A_width,A_height)
        down = count_length(A, x, y, 0, 1,A_width,A_height)
        down_left = count_length(A, x, y, -1, 1,A_width,A_height)
        left = count_length(A, x, y, -1, 0,A_width,A_height)
        left_top = count_length(A, x, y, -1, -1,A_width,A_height)
        horizontal = round(left+right, 2)
        vertical = round(top+down,2)
        diagonal = round(left_top+right_down,2)
        reverse_diagonal = round(top_right+down_left,2)
        min_value=min(horizontal,vertical,diagonal,reverse_diagonal)
        if num_of_crack_pixel==1:
            max_width_x = x
            max_width_y = y
        elif max(crack_width)<min_value:
            max_width_x = x
            max_width_y = y
            max_width_dict['{}_{}'.format(x,y)]=min_value
        crack_width.append(min_value)
    avg_crack_width = sum(crack_width)/len(crack_width)

    return A,avg_crack_width,minx,miny,maxx,maxy,max_width_dict

def visualize_extended_line(A,base_x,base_y,amount_of_change_x,amount_of_change_y,size_x,size_y,color):
    pick_x, pick_y = base_x, base_y

    while A[pick_y, pick_x][0] == 255 :
        A[pick_y, pick_x] = color
        pick_x += amount_of_change_x
        pick_y += amount_of_change_y
        if pick_y == size_y or pick_x == size_x:
            break
        if abs(pick_x - base_x) > 35 or abs(pick_y - base_y) > 35:
            break
    return A

def get_extended_line_x_y(A,base_x,base_y,amount_of_change_x,amount_of_change_y,size_x,size_y):
    pick_x, pick_y = base_x, base_y
    while A[pick_y, pick_x]== 255 :
        pick_x += amount_of_change_x
        pick_y += amount_of_change_y
        if pick_y == size_y :
            return pick_x, pick_y-1
        if pick_x == size_x:
            return pick_x-1, pick_y
        if abs(pick_x - base_x) > 35 or abs(pick_y - base_y) > 35:
            return pick_x,pick_y
    return pick_x,pick_y

def save(A,output_path,fname):
    cv2.imwrite(os.path.join(output_path, fname + '.png'), A)

def binaryize(A,thres):
    A[A > thres] = 255
    A[A <= thres] = 0
    return A

def find_min_direction(A,x,y,width,height):
    top = count_length(A, x, y, 0, -1, width, height)
    top_right = count_length(A, x, y, 1, -1, width,height)
    right = count_length(A, x, y, 1, 0, width,height)
    right_down = count_length(A, x, y, 1, 1, width,height)
    down = count_length(A, x, y, 0, 1, width,height)
    down_left = count_length(A, x, y, -1, 1, width,height)
    left = count_length(A, x, y, -1, 0, width,height)
    left_top = count_length(A, x, y, -1, -1, width,height)
    horizontal = left + right
    vertical = top + down
    diagonal = left_top + right_down
    reverse_diagonal = top_right + down_left
    four_direction_distance = [horizontal, vertical, diagonal, reverse_diagonal]
    min_value = min(four_direction_distance)
    min_idx=four_direction_distance.index(min_value)
    if min_idx == 0:
        return 0 #'horizontal'
    elif min_idx ==1:
        return 1 #'vertical'
    elif min_idx ==2:
        return 2 #'diagonal'
    elif min_idx ==3:
        return 3 #'reverse_diagonal'
    return 'idonknow'

def visualize_circle(A,x_y_color):
    for xyc in x_y_color:
        x=xyc[0]
        y=xyc[1]
        A[y,x]=(0,255,255)
    return A

# def full_process(A):
#
#     A, avg_crack_width, minx, miny, maxx, maxy, max_width_dict = every_search_get_max_avg_width_of_crack(A) # 옛날 함수
#     A_height,A_width= A.shape
#     max_width_x,max_width_y,max_crack_width = get_max_width_x_max_width_y(A,max_width_dict)
#
#     directions=[[-1,0,1,0],[0,-1,0,1],[-1,-1,1,1],[1,-1,-1,1]] ## [ 'horizontal', 'vertical', 'diagonal' , 'reverse_diagonal']
#
#     direction=find_min_direction(A,max_width_x,max_width_y,A_width,A_height)
#     line_x2, line_y2 = get_extended_line_x_y(A, max_width_x, max_width_y, directions[direction][2],directions[direction][3], A_width, A_height)
#     line_x1,line_y1=get_extended_line_x_y(A, max_width_x, max_width_y, directions[direction][0], directions[direction][1], A_width, A_height)
#
#     return max_crack_width, avg_crack_width, max_width_x, max_width_y, minx, miny, maxx, maxy,line_x1,line_y1,line_x2,line_y2

def full_process(A,count,return_list,patch_size,i_j):
    A, total_average_width, minx, miny, maxx, maxy, max_width_dict = every_search_get_max_avg_width_of_crack(A) # 옛날 함수
    if len(A.shape) == 3:
        A_height, A_width, _ = A.shape
    else :
        A_height,A_width= A.shape

    max_width_x,max_width_y,total_max_width = get_max_width_x_max_width_y(A,max_width_dict)
    directions=[[-1,0,1,0],[0,-1,0,1],[-1,-1,1,1],[1,-1,-1,1]] ## [ 'horizontal', 'vertical', 'diagonal' , 'reverse_diagonal']
    direction=find_min_direction(A,max_width_x,max_width_y,A_width,A_height)
    line_x2, line_y2 = get_extended_line_x_y(A, max_width_x, max_width_y, directions[direction][2],directions[direction][3], A_width, A_height)
    line_x1, line_y1 = get_extended_line_x_y(A, max_width_x, max_width_y, directions[direction][0], directions[direction][1], A_width, A_height)

    # print('file name : ', fname)
    # print('----------------------------------------------------------------------------')
    i , j = i_j[0],i_j[1]

    return_list[count] = {
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
                'max_width_y':max_width_y,
                'line_x1':line_x1,
                'line_y1':line_y1,
                'line_x2':line_x2,
                'line_y2':line_y2
            }


def crack_width_analysis(seg_image, threshold, cls_result_data, patch_size=256):
    image = base64.b64decode(seg_image)

    full_img_dict = {}
    input_img = Image.open(BytesIO(image)).convert('L')
    display = np.asarray(input_img)
    display.flags.writeable = True
    display[display <= threshold] = 0
    cv2.imwrite("test_" + str(threshold) + ".png", display)

    width, height = input_img.size
    patch_width = int(width / patch_size)
    patch_height = int(height / patch_size)
    patch_size = 256

    patches = []
    i_j = []
    jobs = []

    for j in tqdm(range(0, patch_height), desc="Divide by Patches"):
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
                        patches.append(patch_img)
                        i_j.append([i, j])

    manager = multiprocessing.Manager()
    full_img_dict = manager.list()
    for i in range(0,len(patches)):
        full_img_dict.append(None)

    for i in tqdm(range(0,len(patches)),desc="Calculating Severity(Multi-processing)"):
            p = multiprocessing.Process(target=full_process, args=(patches[i],i,full_img_dict,patch_size,i_j[i]))
            jobs.append(p)
            p.start()

    for proc in jobs:
        proc.join()

    return full_img_dict
