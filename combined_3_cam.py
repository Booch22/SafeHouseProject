import cv2
import math
import numpy as np
from skimage import color
from skimage import measure



def size_threshold( binary_img : np.ndarray, min_area : float, max_area : float ) :
    obj = 0
    labels = measure.label( binary_img ) # เลเบลวัตถุที่เจอ
    props = measure.regionprops( labels ) # ฟังก์ชันที่ใช้ในการคำนวณคุณสมบัติของวัตถุ 

    # ค้นหาวัตถุและตัดสว่นที่ไม่จำเป็นออกไปจากภาพ
    for prop in props :
        obj += 1
        #print(f'Label: {prop.label} >> Object size: {prop.area}')
        if prop.area > max_area or prop.area < min_area :
            coords = prop.coords[:, ::-1]  # สลับตำแหน่งของ (x, y) เป็น (y, x) ??
            cv2.fillPoly( binary_img, [coords], 0 )  # ถมดำวัตถุที่เข้าเงื่อนไข
    #print( f'Object founded >> {obj}\n' )
            


def check_circularity( check_img ) :
    labels = measure.label( check_img ) # เลเบลวัตถุที่เจอ
    props = measure.regionprops( labels ) # ฟังก์ชันที่ใช้ในการคำนวณคุณสมบัติของวัตถุ 
    contours, _ = cv2.findContours( check_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE ) # หาคอนทัวน์ของวัตถุ

    for prop, contour in zip(props, contours) :
        perimeter = cv2.arcLength(contour, True) # คำนวณความยาวเส้นรอบวงของรูปทรง
        r = ( perimeter / (2 * math.pi) ) + 0.5 # คำนวณรัศมีเฉลี่ย
        circularity_values = ((4 * math.pi * prop.area) / (perimeter ** 2)) * (1 - (0.5 / r)) ** 2  # คำนวณความกลมของวัตถุ

        if circularity_values <= 0.9 :
            coords = prop.coords
            cv2.fillPoly( check_img, [coords], 0 )  # ถมดำวัตถุที่เข้าเงื่อนไข



def bgr_2_cmyk( image : np.ndarray ) :
    ############ # Convert from BGR to CMYK color ################
    # Make float and divide by 255 to give BGRdash

    bgrdash = image.astype( np.float32 )/255.

    # Calculate K as (1 - whatever is biggest out of Rdash, Gdash, Bdash)
    K = 1 - np.max(bgrdash, axis=2)
    # Calculate C
    C = (1-bgrdash[...,2] - K)/(1-K) #use Red chanel
    # Calculate M
    M = (1-bgrdash[...,1] - K)/(1-K) #use Green chanel
    # Calculate Y
    Y = (1-bgrdash[...,0] - K)/(1-K) #use Blue chanel
    # Combine 4 channels into single image and re-scale back up to uint8
    CMYK = (np.dstack((C,M,Y,K)) * 255).astype(np.uint8)
    return CMYK



def detect_object( image_for_draw_rec : np.ndarray, target_img : np.ndarray ) :
    # หา contours ของวัตถุ โดยค่าที่ได้จะเป็นจุดเชื่อมของวัตถุ
    contours, _ = cv2.findContours( target_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    centroid_x = []
    centroid_y = []

    # วาดกรอบสี่เหลี่ยมรอบวัตถุ
    centr = 0 # ใช้ในการเลื่อนลำดับข้อมูลที่เก็บใน centroid
    for contour in contours:
        x, y, w, h = cv2.boundingRect( contour ) # หากกรอบสี่เหลี่ยมเพื่อตรวจจับวัตถุ
        if (w / h) >= 0.8 and (w / h) <= 1.3   : # หาความสมมาตรของวัตถุ เพราะถ้าเป็นหมุดจะมีความกว้างและความยาวเท่ากัน
            #print( f"w / h = {w / h}" )
            centroid_x.append( x + int( (w / 2) ) ) # เก็บตำแหน่งตรงกลางของแกน x
            centroid_y.append( y + int( (h / 2) ) ) # เก็บตำแหน่งตรงกลางของแกน y
            cv2.rectangle( image_for_draw_rec, (x, y), (x + w, y + h), (0, 255, 0), 2 ) # ตีกรอบสี่เหลี่ยม
            cv2.drawMarker( image_for_draw_rec, (centroid_x[centr], centroid_y[centr]), (0, 255, 0), markerType = cv2.MARKER_TILTED_CROSS, markerSize = 5, thickness = 1 ) # มาร์กจุดกลาง
            centr += 1
    return centroid_x, centroid_y



def calculate_distance( img_for_show_distance : np.ndarray, centroid_x : list, centroid_y : list, 
                        ruler_millimeter : float, ruler_pixels : float ) :
    # check
    if len( centroid_x ) >= 2  and len( centroid_y ) >= 2 :
        x1 = centroid_x[0] # centroid แกน x ตำแหน่งที่ 1
        y1 = centroid_y[0] # centroid แกน y ตำแหน่งที่ 1
        x2 = centroid_x[1] # centroid แกน x ตำแหน่งที่ 2
        y2 = centroid_y[1] # centroid แกน y ตำแหน่งที่ 2
        text_position = ( int(((x1 + x2) / 2) + 5), int(((y1 + y2) / 2) + 5) ) # ตำแหน่งที่จะแสดงค่า distance

        if ruler_pixels > 0 : # หากเจอเหรียญให้คำนวณ
            # คำนวณระยะห่างระหว่างหมุด (euclidean_distance)
            euclidean_value = math.sqrt( (x1 - x2)**2 + (y1 - y2)**2 )
            # แปลง pixels เป็น millimeter
            #print( f'( {euclidean_value} * {ruler_millimeter} ) / {ruler_pixels}' )
            distance_value = ( euclidean_value * ruler_millimeter ) / ruler_pixels
            # แสดงค่าระยะห่างของหมุด
            cv2.putText( img_for_show_distance, f"{round(distance_value, 3)}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255) )
            # ลากเส้นระยะของหมุด
            cv2.line( img_for_show_distance, (centroid_x[0], centroid_y[0]), (centroid_x[1], centroid_y[1]), color = (0, 255, 0), thickness = 2 )
            
        elif ruler_pixels == 0 : # หากไม่เจอเหรียญ
            print( "Need diameter of coin for calculate distance" )



def marker( select_cam : int ) :

    # คำนวณระยะที่วัดได้จากไม้บรรทัด โดยค่าที่ได้จะมีหน่วยเป็น pixels
    def ruler_distance( img_for_show_distance : np.ndarray, point_1 : tuple, point_2 : tuple ) :
        # check
        x1 = point_1[0] # centroid แกน x ตำแหน่งที่ 1
        y1 = point_1[1] # centroid แกน y ตำแหน่งที่ 1
        x2 = point_2[0] # centroid แกน x ตำแหน่งที่ 2
        y2 = point_2[1] # centroid แกน y ตำแหน่งที่ 2
        text_position = ( int((x1 + x2) / 2), int(((y1 + y2) / 2) + 15) ) # ตำแหน่งที่จะแสดงค่า distance
        # คำนวณระยะห่างระหว่างไม้บรรทัด (euclidean_distance)
        ruler_pixels.append( math.sqrt( (x1 - x2)**2 + (y1 - y2)**2 ) )
        cv2.putText( img_for_show_distance, f"{round(ruler_pixels[0], 3)}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0) )

    # มาร์กตำแหน่งของระยะไม้บรรทัด
    def mark_position( event, x, y, flags, param ) :
        # เมื่อคลิกซ้าย
        if event == cv2.EVENT_LBUTTONDOWN : 
            # คลิกได้ 2 ครั้ง
            if len( clicked_position ) < 2 :
                clicked_position.append( (x, y) )
                cv2.drawMarker( marker_img, (x, y), (0, 0, 0), markerType = cv2.MARKER_TILTED_CROSS, markerSize = 10, thickness = 1 ) # มาร์กจุดกลาง
                cv2.imshow( "marker image", marker_img )
                # หากมาร์กครบ 2 ตำแหน่งให้ลากเส้น
                if len( clicked_position ) == 2 :
                    ruler_distance( marker_img, clicked_position[0], clicked_position[1] ) # คำนวณระยะห่างระหว่างจุดมาร์ก
                    cv2.line( marker_img, clicked_position[0], clicked_position[1], color = (0, 0, 0), thickness = 1 )
                    cv2.imshow( "marker image", marker_img )


    ruler_milimeter = 10 # กำหนดความยาวของไม้บรรทัดเป็น 20 mm.
    ruler_pixels = [] # ความยาวของไม้บรรทัด เก็บเป็น pixels
    clicked_position = [] # เก็บตำแหน่ง x, y

    # นำเข้ารูปภาพ
    if select_cam == 1 : # มุมกล้อง 1
        image = cv2.imread( "Benchmark_Normal_Frames/frame288.jpg" ) # 288
        re_img = cv2.resize( image, (1536, 864) ) # resize
        marker_img = re_img[region_1[1]:region_1[1]+region_1[3], region_1[0]:region_1[0]+region_1[2]]
    elif select_cam == 2 : # มุมกล้อง 2
        image = cv2.imread( "Benchmark_Normal_Frames/frame332.jpg" ) # 332
        re_img = cv2.resize( image, (1536, 864) ) # resize
        marker_img = re_img[region_2[1]:region_2[1]+region_2[3], region_2[0]:region_2[0]+region_2[2]]
    elif select_cam == 3 : # มุมกล้อง 3
        image = cv2.imread( "Benchmark_Normal_Frames/frame436.jpg" ) # 436
        re_img = cv2.resize( image, (1536, 864) ) # resize
        marker_img = re_img[region_3[1]:region_3[1]+region_3[3], region_3[0]:region_3[0]+region_3[2]]

    # แสดงรูปภาพก่อนแปลง
    cv2.imshow( "marker image", marker_img ) 
    # ทำงานกับเมาส์
    cv2.setMouseCallback( "marker image", mark_position )
    # แสดงรูปภาพค้างไว้
    cv2.waitKey(0)
    # Window shown waits for any key pressing event 
    cv2.destroyAllWindows()

    print( f'ruler mm = {ruler_milimeter}, ruler px = {ruler_pixels[0]}' )
    return ruler_milimeter, ruler_pixels[0]



# กำหนดตำแหน่งและขนาดของภาพย่อย
# ประกอบไปด้วย ( x, y, width, height )
region_1 = ( 384, 0, 576, 432 )
region_2 = ( 960, 0, 576, 432 )
region_3 = ( 960, 432, 576, 432 )


# มาร์กตำแหน่งสำหรับไม้บรรทัด
ruler_mill_1, ruler_pix_1 = marker( 1 ) # cam 1
ruler_mill_2, ruler_pix_2 = marker( 2 ) # cam 2
ruler_mill_3, ruler_pix_3 = marker( 3 ) # cam 3


# กำหนดค่า threshold
black_threshold = 155
green_threshold = 134
blue_threshold = { 'cam 1' : 120, 'cam 2' : 124, 'cam 3' : 124 }


# กำหนดเฟรมเริ่มต้นของรูปภาพ
start_frame = 0
# กำหนดขนาดรูปที่จะแสดง
re_w, re_h = 1536, 864
# สร้างหน้าต่างสำหรับ morphology operation
kernel = np.ones( (5, 5), np.uint8 )
for frame in range( 1500 ) :
    # โหลดภาพ
    # เส้นทางของรูปภาพที่จะแสดง
    path_of_frame = 'Benchmark_Normal_Frames/' + 'frame' +  str(start_frame) +'.jpg'
    print( path_of_frame )
    input_image = cv2.imread( path_of_frame )

    # ย่อขนาดรูปภาพ
    resized_image = cv2.resize( input_image, (re_w, re_h) )


    # สร้างภาพย่อยจากภาพหลัก
    # โดยระบุตำแหน่งจากอาร์เรย์ของภาพหลัก [ y : y + height, x : x + width ]
    sub_image_1 = resized_image[region_1[1]:region_1[1]+region_1[3], region_1[0]:region_1[0]+region_1[2]]
    sub_image_2 = resized_image[region_2[1]:region_2[1]+region_2[3], region_2[0]:region_2[0]+region_2[2]]
    sub_image_3 = resized_image[region_3[1]:region_3[1]+region_3[3], region_3[0]:region_3[0]+region_3[2]]

 
    # แยกสีหมุดด้วยการใช้ภาพแบบ LAB และ CMYK
    # cam 1 (LAB)
    lab_img_1 = cv2.cvtColor( sub_image_1, cv2.COLOR_BGR2LAB )
    # cam 2 (LAB)
    lab_img_2 = cv2.cvtColor( sub_image_2, cv2.COLOR_BGR2LAB )
    # cam 3 (LAB, CMYK)
    lab_img_3 = cv2.cvtColor( sub_image_3, cv2.COLOR_BGR2LAB )
    cmyk_img_3 = bgr_2_cmyk( sub_image_3 )


    # แยกแต่ละชาแนล เพื่อนำชาแนลที่เหมาะสมมาใช้งาน
    # cam 1
    l_img_1, a_img_1, b_img_1 = cv2.split( lab_img_1 )
    # cam 2
    l_img_2, a_img_2, b_img_2 = cv2.split( lab_img_2 )
    # cam 3
    l_img_3, a_img_3, b_img_3 = cv2.split( lab_img_3 )
    c_img_3, m_img_3, y_img_3, k_img_3 = cv2.split( cmyk_img_3 )

    
    # ปรับภาพเกรย์สเกลด้วย Gaussian Blur
    # cam 1
    blur_green_img_1 = cv2.GaussianBlur( a_img_1, (5, 5), 10 )
    blur_blue_img_1 = cv2.GaussianBlur( b_img_1, (5, 5), 10 )
    # cam 2
    blur_blue_img_2 = cv2.GaussianBlur( b_img_2, (5, 5), 10 )
    # cam 3
    blur_blue_img_3 = cv2.GaussianBlur( b_img_3, (5, 5), 10 )
    blur_black_img_3 = cv2.GaussianBlur( k_img_3, (5, 5), 10 )


    # ทำการ thresholding
    # cam 1
    ret_a_1, thresh_green_img_1 = cv2.threshold( blur_green_img_1, green_threshold, 255, cv2.THRESH_BINARY_INV )
    ret_b_1, thresh_blue_img_1 = cv2.threshold( blur_blue_img_1, blue_threshold['cam 1'], 255, cv2.THRESH_BINARY_INV )
    # cam 2
    ret_b_2, thresh_blue_img_2 = cv2.threshold( blur_blue_img_2, blue_threshold['cam 2'], 255, cv2.THRESH_BINARY_INV )
    # cam 3
    ret_k_3, thresh_black_img_3 = cv2.threshold( blur_black_img_3, black_threshold, 255, cv2.THRESH_BINARY )
    ret_b_3, thresh_blue_img_3 = cv2.threshold( blur_blue_img_3, blue_threshold['cam 3'], 255, cv2.THRESH_BINARY_INV )

    
    # morphology operation
    # cam 1
    opening_green_img_1 = cv2.morphologyEx( thresh_green_img_1, cv2.MORPH_OPEN, kernel )
    closing_green_img_1 = cv2.morphologyEx( opening_green_img_1, cv2.MORPH_CLOSE, kernel )
    opening_blue_img_1 = cv2.morphologyEx( thresh_blue_img_1, cv2.MORPH_OPEN, kernel )
    closing_blue_img_1 = cv2.morphologyEx( opening_blue_img_1, cv2.MORPH_CLOSE, kernel )
    # cam 2
    opening_blue_img_2 = cv2.morphologyEx( thresh_blue_img_2, cv2.MORPH_OPEN, kernel )
    closing_blue_img_2 = cv2.morphologyEx( opening_blue_img_2, cv2.MORPH_CLOSE, kernel )
    # cam 3
    opening_blue_img_3 = cv2.morphologyEx( thresh_blue_img_3, cv2.MORPH_OPEN, kernel )
    closing_blue_img_3 = cv2.morphologyEx( opening_blue_img_3, cv2.MORPH_CLOSE, kernel )
    opening_black_img_3 = cv2.morphologyEx( thresh_black_img_3, cv2.MORPH_OPEN, kernel )
    closing_black_img_3 = cv2.morphologyEx( opening_black_img_3, cv2.MORPH_CLOSE, kernel )



    # size threshold
    # cam 1
    size_threshold( closing_green_img_1, 150, 250 ) # นับวัตถุที่เป็นสีขาว
    size_threshold( closing_blue_img_1, 80, 250 ) # นับวัตถุที่เป็นสีขาว
    # cam 2
    size_threshold( closing_blue_img_2, 150, 350 )
    # cam 3
    size_threshold( closing_black_img_3, 250, 600 ) # นับวัตถุที่เป็นสีขาว
    size_threshold( closing_blue_img_3, 300, 700 ) # นับวัตถุที่เป็นสีขาว
    
    
    # ตรวจความกลมของวัตถุ หากวัตถุไม่กลมให้ถมดำ
    # cam 1
    check_circularity( closing_green_img_1 )
    check_circularity( closing_blue_img_1 )
    # cam 2
    check_circularity( closing_blue_img_2 )
    # cam 3
    check_circularity( closing_black_img_3 )
    check_circularity( closing_blue_img_3 )
    
    
    # ตรวจจับหมุดและเหรียญ
    # cam 1
    centr_x_green_1, centr_y_green_1 = detect_object( sub_image_1, closing_green_img_1 )
    centr_x_blue_1, centr_y_blue_1 = detect_object( sub_image_1, closing_blue_img_1 )
    # cam 2
    centr_x_blue_2, centr_y_blue_2 = detect_object( sub_image_2, closing_blue_img_2 )
    # cam 3
    centr_x_black_3, centr_y_black_3 = detect_object( sub_image_3, closing_black_img_3 )
    centr_x_blue_3, centr_y_blue_3 = detect_object( sub_image_3, closing_blue_img_3 )


    # วัดระยะของหมุด
    # cam 1
    calculate_distance( sub_image_1, centr_x_blue_1, centr_y_blue_1, ruler_mill_1, ruler_pix_1 )
    calculate_distance( sub_image_1, centr_x_green_1, centr_y_green_1, ruler_mill_1, ruler_pix_1 )
    # cam 2
    calculate_distance( sub_image_2, centr_x_blue_2, centr_y_blue_2, ruler_mill_2, ruler_pix_2 )
    # cam 3
    calculate_distance( sub_image_3, centr_x_blue_3, centr_y_blue_3, ruler_mill_3, ruler_pix_3 )
    calculate_distance( sub_image_3, centr_x_black_3, centr_y_black_3, ruler_mill_3, ruler_pix_3 )
    

    # แสดงภาพที่ได้จาก Image Thresholding
    #cv2.imshow( "threshold image", thresh_blue_img_3 )
    #cv2.imshow( "binary image", closing_blue_img_3 )
    cv2.imshow( "main image", resized_image )
    cv2.waitKey( 10 ) # 220
    start_frame += 2 

cv2.destroyAllWindows()
