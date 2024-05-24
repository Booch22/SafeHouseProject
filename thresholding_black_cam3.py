import cv2
import math
import numpy as np
from skimage import color
from skimage import measure
import matplotlib.pyplot as plt



def show_histogram( img : np.ndarray ) :
    hist = cv2.calcHist( [img], [0], None, [256], [0, 256] )
    # แสดง histogram ด้วย Matplotlib
    plt.plot(hist)
    plt.title("Gray Scale Histogram")
    plt.xlabel("Gray Value")
    plt.ylabel("Frequency")
    plt.show()



def show_histogram_float( img ):
    # คำนวณ histogram
    hist, bins = np.histogram(img, bins=256, range=(-1, 1))
    # แสดง histogram ด้วย Matplotlib
    plt.plot(bins[:-1], hist)
    plt.title("Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.show()



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


        # เงื่อนไขสำหรับใช้เลือกวัตถุที่มีความกลม
        if circularity_values <= 0.6 :
            coords = prop.coords[:, ::-1]  # สลับตำแหน่งของ (x, y) เป็น (y, x) ??
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

        print( f"w / h = {w / h}" )
        if (w / h) >= 0.85 and (w / h) <= 1.2 : # หาความสมมาตรของวัตถุ เพราะถ้าเป็นหมุดจะมีความกว้างและความยาวเท่ากัน
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
        text_position = ( int((x1 + x2) / 2), int(((y1 + y2) / 2) + 15) ) # ตำแหน่งที่จะแสดงค่า distance

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
                cv2.drawMarker( marker_img, (x, y), (0, 0, 0), markerType = cv2.MARKER_TILTED_CROSS, markerSize = 10, thickness = 2 ) # มาร์กจุดกลาง
                cv2.imshow( "marker image", marker_img )
                # หากมาร์กครบ 2 ตำแหน่งให้ลากเส้น
                if len( clicked_position ) == 2 :
                    ruler_distance( marker_img, clicked_position[0], clicked_position[1] ) # คำนวณระยะห่างระหว่างจุดมาร์ก
                    cv2.line( marker_img, clicked_position[0], clicked_position[1], color = (0, 0, 0), thickness = 2 )
                    cv2.imshow( "marker image", marker_img )


    ruler_milimeter = 10 # กำหนดความยาวของไม้บรรทัดเป็น 20 mm.
    ruler_pixels = [] # ความยาวของไม้บรรทัด เก็บเป็น pixels
    clicked_position = [] # เก็บตำแหน่ง x, y

    # นำเข้ารูปภาพ
    image = cv2.imread( "Benchmark_Normal_Frames/frame436.jpg" ) # cam 3 = 436

    # แบ่งรูปภาพย่อย 
    sub_image_1 = image[region_1[1]:region_1[1]+region_1[3], region_1[0]:region_1[0]+region_1[2]]
    sub_image_2 = image[region_2[1]:region_2[1]+region_2[3], region_2[0]:region_2[0]+region_2[2]]
    sub_image_3 = image[region_3[1]:region_3[1]+region_3[3], region_3[0]:region_3[0]+region_3[2]]

    # เลือกมุมกล้อง
    if select_cam == 1 :
        marker_img = sub_image_1
    elif select_cam == 2 :
        marker_img = sub_image_2
    elif select_cam == 3 :
        marker_img = sub_image_3

    # แสดงรูปภาพก่อนแปลง
    cv2.imshow( "marker image", marker_img ) 
    # ทำงานกับเมาส์
    cv2.setMouseCallback( "marker image", mark_position )
    # แสดงรูปภาพค้างไว้
    cv2.waitKey(0)
    # Window shown waits for any key pressing event 
    cv2.destroyAllWindows()

    return ruler_milimeter, ruler_pixels[0]



# กำหนดตำแหน่งและขนาดของภาพย่อย
# ประกอบไปด้วย ( x, y, width, height )
region_1 = (481, 1, 719, 539)
region_2 = (1201, 1, 719, 539)
region_3 = (1201, 541, 719, 539)


# มาร์กตำแหน่งสำหรับไม้บรรทัด
ruler_mill, ruler_pix = marker( 3 )
print( f"{ruler_mill}, {ruler_pix}" )


# กำหนดค่า threshold
b_threshold = 124 # น้ำเงิน
k_threshold = 155 # ดำ 145
coin_threshold = 115


# กำหนดเฟรมเริ่มต้นของรูปภาพ
start_frame = 0
for frame in range( 1500 ) :
    # โหลดภาพ
    # เส้นทางของรูปภาพที่จะแสดง
    path_of_frame = 'Benchmark_Normal_Frames/' + 'frame' +  str(start_frame) +'.jpg'
    #path_of_frame = 'data_pool_3_right/' + 'frame' +  str(start_frame) +'.jpg'


    print( path_of_frame )
    input_image = cv2.imread( path_of_frame )

    # สร้างภาพย่อยจากภาพหลัก
    # โดยระบุตำแหน่งจากอาร์เรย์ของภาพหลัก [ y : y + height, x : x + width ]
    sub_image_3 = input_image[region_3[1]:region_3[1]+region_3[3], region_3[0]:region_3[0]+region_3[2]]


    # ภาพหลัก
    main_img = sub_image_3.copy()


    # แปลง bgr เป็น rgb
    rgb_img = cv2.cvtColor( main_img, cv2.COLOR_BGR2RGB )
    # แยกแต่ละชาแนลออกจากกัน
    r_img, g_img, b_img = cv2.split( rgb_img )


    # แปลงจาก BGR เป็น LAB
    lab_img = cv2.cvtColor( main_img, cv2.COLOR_BGR2LAB )
    # น้ำเงินใช้ B
    l_img, a_img, b_img = cv2.split( lab_img )


    # แปลง rgb เป็น cmyk
    cmyk_img = bgr_2_cmyk( main_img )
    # แยกแต่ละชาแนลจาก cmyk
    c_img, m_img, y_img, k_img = cv2.split( cmyk_img )


    # show histogram of image
    #show_histogram( b_img )
    #show_histogram_float( i_img )


    # ปรับภาพเกรย์สเกลด้วย Gaussian Blur
    gauss_blur_k_img = cv2.GaussianBlur( k_img, (9, 9), 10 )
    gauss_blur_b_img = cv2.GaussianBlur( b_img, (9, 9), 10 )
    

    # ทำการ thresholding
    ret_k, thresh_k_img = cv2.threshold( gauss_blur_k_img, k_threshold, 255, cv2.THRESH_BINARY )
    ret_b, thresh_b_img = cv2.threshold( gauss_blur_b_img, b_threshold, 255, cv2.THRESH_BINARY_INV )


    #cv2.imshow( "thresh image", thresh_k_img )


    # morphology operation
    kernel = np.ones( (9, 9), np.uint8 ) # สร้างหน้าต่าง
    # morphology ดำ
    opening_k_img = cv2.morphologyEx( thresh_k_img, cv2.MORPH_OPEN, kernel )
    closing_k_img = cv2.morphologyEx( opening_k_img, cv2.MORPH_CLOSE, kernel )
    # น้ำเงิน
    opening_b_img = cv2.morphologyEx( thresh_b_img, cv2.MORPH_OPEN, kernel )
    closing_b_img = cv2.morphologyEx( opening_b_img, cv2.MORPH_CLOSE, kernel )    


    # size threshold
    # ตัดส่วนที่ไม่จำเป็นออกไปจากภาพ เหลือแต่ส่วนที่ต้องการ
    size_threshold( closing_k_img, 400, 1100 ) # นับวัตถุที่เป็นสีขาว
    size_threshold( closing_b_img, 400, 1000 ) # นับวัตถุที่เป็นสีขาว


    # ตรวจความกลมของวัตถุ หากวัตถุไม่กลมให้ถมดำ
    #check_circularity( closing_k_img )
    #check_circularity( closing_b_img )

    # ตรวจจับหมุดและเหรียญ
    centr_x_black, centr_y_black = detect_object( main_img, closing_k_img )
    centr_x_blue, centr_y_blue = detect_object( main_img, closing_b_img )


    # วัดระยะของหมุด
    calculate_distance( main_img, centr_x_black, centr_y_black, ruler_mill, ruler_pix )
    calculate_distance( main_img, centr_x_blue, centr_y_blue, ruler_mill, ruler_pix )

    # แสดงภาพที่ได้จาก Image Thresholding
    cv2.imshow( "main image", main_img )
    cv2.imshow( "k image", closing_k_img )
    cv2.waitKey( 100 ) # 220
    start_frame += 2

cv2.destroyAllWindows()