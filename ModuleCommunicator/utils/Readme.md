# region.py
패치 단위로 분류된 결과를 영역으로 묶는 작업을 하는 코드
- input : 패치별 분류결과
- output : 영역별 분류결과

## make_region 함수
region.py의 main함수로 패치별 분류결과를 받아서 영역별 분류결과를 출력한다.
- input : image_path, segmentation된 pothole 이미지, 패치별 분류결과, image_width, image_height, patch_size, region_threshold(몇개부터 region으로), connectivity_option(4방향, 8방향), noise_filtering_option(구멍 매울것인가, 주변을 확인하여 채울것인가)
     
- output : 영역별 분류결과

## region_numbering 함수
패치 단위 결과를 주변 패치에 따라 영역 번호를 매기는 함수, connectivity_option에 의해 4방향으로 할지 8방향으로 할지 선택한다.
- input : crack_map(3704x10000를 256x256으로 나눈 14x39의 파손패치 지도), axis_x, axis_y, connectivity_option
- output : numbering된 crack_map

## check_nearby_region 함수
이중 for loop가 우측, 아래로 패치 분류 결과를 확인하며 영역 번호를 매겨 겹친 부분에 대한 처리를 하기 위한 함수
- input : crack_map, axis_x, axis_y, connectivity_option
- output : numbering되고 주변 같은 영역까지 처리된 crack_map

## noise_filtering 함수
영역별로 묶인 결과중 패치별 결과에 구멍이 있거나 주변의 분류 결과를 토대로 noise filtering 하는 함수
- input : crack_map, axis_x, axis_y, noise_filtering_option
- output : noise filtering된 crack_map

## crack_region_severity 함수
균열의 심각도를 입력되는 최대폭에 의해 low, medium, high로 나누는 함수
- input : max_of_total_max_width
- output : severity(low, medium, high)

## patching_region_severity 함수
patching영역의 넓이를 구하기 위한 함수
- input : input_image(gray이미지), patching_region, patch_size
- output : area(넓이), bbox, contour(외곽선), seg_image(patching 픽셀만 segmentation된 이미지)

## pothole_region_severity 함수
pothole영역의 넓이를 구하기 위한 함수
- input : input_image, distress_region, patch_size
- output : area, bbox, contour

## region_thresholding 함수
일정 패치 갯수 이하의 region을 없애버리는 함수
- input : region_result(region 분류 결과), region_threshold
- output : region_result

## distress_thresholding 함수
파손 형태별 파손의 최소 요건을 만족하지 못하면 해당 region을 없애버리는 함수(ex: patching을 넓이 150mm이하..)
- input : region_result, pothole_threshold, patching_region_threshold
- output : region_result
## crack_length_area 함수 
균열의 최대폭 혹은 넓이를 계산하는 함수
- input : region_type, crack_bbox
- output : length, area


# severity.py
균열의 심각도를 구하는 작업의 코드 