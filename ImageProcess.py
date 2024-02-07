import cv2
import sys
import math
import numpy as np
from skimage import measure


class processing():
    def __init__(self):
        # กำหนดตำแหน่งและขนาดของภาพย่อย
        # ประกอบไปด้วย ( x, y, width, height )
        self.region_1 = ( 481, 1, 719, 539 )
        self.region_2 = ( 1201, 1, 719, 539 )
        self.region_3 = ( 1201, 541, 719, 539 )
        
        # สำหรับประมวลผล 
        # ลดขนาดรูปภาพลง 10%
        ten_of_w = int( 719 * 0.1 )
        ten_of_h = int( 539 * 0.1 )
        self.pro_region_1 = (  481 + ten_of_w,   1 + ten_of_h, 719 - (ten_of_w + ten_of_w), 539 - (ten_of_h + ten_of_h) )
        self.pro_region_2 = ( 1201 + ten_of_w,   1 + ten_of_h, 719 - (ten_of_w + ten_of_w), 539 - (ten_of_h + ten_of_h) )
        self.pro_region_3 = ( 1201 + ten_of_w, 541 + ten_of_h, 719 - (ten_of_w + ten_of_w), 539 - (ten_of_h + ten_of_h) )
        
        # กำหนดค่า threshold
        self.black_threshold = 155
        self.green_threshold = 134
        self.blue_threshold = { 'cam 1' : 120, 'cam 2' : 124, 'cam 3' : 124 }
        self.semi_threshold = 135 # threshold สำหรับ a* เพื่อให้เห็น มือ ไม้โปร และเวอร์เนียร์



    # ถมดำวัตถุที่ใหญ่เกินไปหรือเล็กเกินไป พร้อมทั้งตรวจจับมือ เวอร์เนียร์ หรือไม้บรรทัด
    def size_threshold( self, binary_img : np.ndarray, min_area : float, max_area : float, 
                        threshold_area : int, semi : bool ) :
        obj = 0 # ใช้นับวัตถุที่เจอ
        labels = measure.label( binary_img ) # เลเบลวัตถุที่เจอ
        props = measure.regionprops( labels ) # ฟังก์ชันที่ใช้ในการคำนวณคุณสมบัติของวัตถุ 
        obj_area = [] # เก็บค่าพื้นที่ของแต่ละวัตถุ
        sum_all_area = 0 # ใช้รวมค่าทั้งหมดของ obj_area
        detect_decision = True # เงื่อนไขตัดสินใจในการตรวจจับ

        # ค้นหาวัตถุและตัดส่วนที่ไม่จำเป็นออกไปจากภาพ
        for prop in props :
            obj += 1
            # ถมดำวัตถุที่เข้าเงื่อนไข
            if semi == False :
                if prop.area > max_area or prop.area < min_area :
                    coords = prop.coords[:, ::-1]  # สลับตำแหน่งของ (x, y) เป็น (y, x) ??
                    cv2.fillPoly( binary_img, [coords], 0 )  # ถมดำ
            # ตรวจจับมือและเวอร์เนียร์
            elif semi == True :
                obj_area.append( prop.area )

        # รวมค่าพื้นที่ทั้งหมดของวัตถุ
        sum_all_area = sum( obj_area )
        # หากค่าความต่างเกิน ไม่ต้องตรวจจับ
        if sum_all_area > threshold_area : # 107000
            detect_decision = False
        
        return detect_decision



    # เช็คความกลมของวัตถุ หากไม่กลมให้ถมดำ
    def check_circularity( self, check_img ) :
        labels = measure.label( check_img ) # เลเบลวัตถุที่เจอ
        props = measure.regionprops( labels ) # ฟังก์ชันที่ใช้ในการคำนวณคุณสมบัติของวัตถุ 
        contours, _ = cv2.findContours( check_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE ) # หาคอนทัวน์ของวัตถุ

        for prop, contour in zip(props, contours) :
            x, y, w, h = cv2.boundingRect( contour ) # หากกรอบสี่เหลี่ยมเพื่อตรวจจับวัตถุ
            perimeter = cv2.arcLength(contour, True) # คำนวณความยาวเส้นรอบวงของรูปทรง
            r = ( perimeter / (2 * math.pi) ) + 0.5 # คำนวณรัศมีเฉลี่ย
            circularity_values = ((4 * math.pi * prop.area) / (perimeter ** 2)) * (1 - (0.5 / r)) ** 2  # คำนวณความกลมของวัตถุ
            # เงื่อนไขสำหรับการลบวัตถุที่ไม่มีความกลม
            if circularity_values <= 0.65 :
                delete_position = np.array( [[x, y], [x, y+h,], [x+w, y+h], [x+w, y]] ) # ตำแหน่งของวัตถุ
                cv2.fillPoly( check_img, pts = [delete_position], color = (0, 0, 0) )  # ถมดำวัตถุที่เข้าเงื่อนไข



    # แปลงภาพสีเป็นภาพcmyk
    def bgr_2_cmyk(self, InputImage : np.ndarray ) :
        # เนื่องจากข้อมูลที่เข้ามาเป็น RGB ต้องแปลงเป็น BGR ก่อน
        image = cv2.cvtColor( InputImage, cv2.COLOR_RGB2BGR )

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



    # ตีกรอบวัตถุที่เจอ
    def detect_object( self, image_for_draw_rec : np.ndarray, target_img : np.ndarray, camera : int ) :
        # หา contours ของวัตถุ โดยค่าที่ได้จะเป็นจุดเชื่อมของวัตถุ
        contours, _ = cv2.findContours( target_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
        centroid_x = [] # เก็บตำแหน่งตรงกลางของแกน x
        centroid_y = [] # เก็บตำแหน่งตรงกลางของแกน y
        object_info = { 'x' : [], 'y' : [], 'w' : [], 'h' : [] } # เก็บค่าหมุดที่เจอ เพื่อนำไปใช้ในการตรวจจับผ่านหน้าจอ

        # วาดกรอบสี่เหลี่ยมรอบวัตถุ
        centr = 0 # ใช้ในการเลื่อนลำดับข้อมูลที่เก็บใน centroid
        for contour in contours:
             # หากกรอบสี่เหลี่ยมเพื่อตรวจจับวัตถุ
            x, y, w, h = cv2.boundingRect( contour )
            # หาความสมมาตรของวัตถุ เพราะถ้าเป็นหมุดจะมีความกว้างและความยาวเท่ากัน
            if (w / h) >= 0.85 and (w / h) < 1.3 : 
                # ค่า x และ y ต้องนำไปรวมกับตำแหน่งของแต่ละมุมกล้อง จึงจะได้ตำแหน่งที่ถูกต้อง
                if camera == 1 :
                    #เก็บตำแหน่ง x, y ของวัตถุ
                    object_info['x'].append( x + self.pro_region_1[0] )
                    object_info['y'].append( y + self.pro_region_1[1] )
                    #เก็บตำแหน่งตรงกลางของแต่ละวัตถุ
                    centroid_x.append( (x + self.pro_region_1[0]) + int( (w / 2) ) )
                    centroid_y.append( (y + self.pro_region_1[1]) + int( (h / 2) ) )
                elif camera == 2 :
                    #เก็บตำแหน่ง x, y ของวัตถุ
                    object_info['x'].append( x + self.pro_region_2[0] )
                    object_info['y'].append( y + self.pro_region_2[1] )
                    #เก็บตำแหน่งตรงกลางของแต่ละวัตถุ
                    centroid_x.append( (x + self.pro_region_2[0]) + int( (w / 2) ) )
                    centroid_y.append( (y + self.pro_region_2[1]) + int( (h / 2) ) )
                elif camera == 3 :
                    #เก็บตำแหน่ง x, y ของวัตถุ
                    object_info['x'].append( x + self.pro_region_3[0] )
                    object_info['y'].append( y + self.pro_region_3[1] )
                    #เก็บตำแหน่งตรงกลางของแต่ละวัตถุ
                    centroid_x.append( (x + self.pro_region_3[0]) + int( (w / 2) ) )
                    centroid_y.append( (y + self.pro_region_3[1]) + int( (h / 2) ) )

                object_info['w'].append( w )
                object_info['h'].append( h )
                centr += 1

        return object_info, centroid_x, centroid_y



    # คำนวณระยะห่างระหว่างจุดสองจุด
    def calculate_distance( self, img_for_show_distance : np.ndarray, centroid_x : list, centroid_y : list, 
                            ruler_millimeter : float, ruler_pixels : float ) :
        # สร้างตัวแปรเก็บค่ากึ่งกลางสำหรับลากเส้น
        distance_info = { 'x1' : 0, 'y1' : 0, 'x2' : 0, 'y2' : 0, 'distance' : 0, 'text position' : [0, 0] }

        # check
        if len( centroid_x ) >= 2  and len( centroid_y ) >= 2 :
            distance_info['x1'] = centroid_x[0] # centroid แกน x ตำแหน่งที่ 1
            distance_info['y1'] = centroid_y[0] # centroid แกน y ตำแหน่งที่ 1
            distance_info['x2'] = centroid_x[1] # centroid แกน x ตำแหน่งที่ 2
            distance_info['y2'] = centroid_y[1] # centroid แกน y ตำแหน่งที่ 2
            distance_info['text position'] = [ int(((centroid_x[0] + centroid_x[1]) / 2) + 5), int(((centroid_y[0] + centroid_y[1]) / 2) + 5) ] # ตำแหน่งที่จะแสดงค่า distance

            if ruler_pixels > 0 : # หากเจอเหรียญให้คำนวณ
                # คำนวณระยะห่างระหว่างหมุด (euclidean_distance)
                euclidean_value = math.sqrt( (centroid_x[0] - centroid_x[1])**2 + (centroid_y[0] - centroid_y[1])**2 )
                # แปลง pixels เป็น millimeter
                distance_info['distance'] = ( euclidean_value * ruler_millimeter ) / ruler_pixels
            
            elif ruler_pixels == 0 : # หากไม่เจอเหรียญ
                print( "Need diameter of coin for calculate distance" )

        return distance_info


    # ฟังก์ชั่นสำหรับมาร์คตำแหน่งและเก็บค่าไม้บรรทัด
    def marker(self, select_cam, frame) :
        # คำนวณระยะที่วัดได้จากไม้บรรทัด โดยค่าที่ได้จะมีหน่วยเป็น pixels
        def ruler_distance( img_for_show_distance : np.ndarray, point_1 : tuple, point_2 : tuple ) :
            # check
            x1 = point_1[0] # centroid แกน x ตำแหน่งที่ 1
            y1 = point_1[1] # centroid แกน y ตำแหน่งที่ 1
            x2 = point_2[0] # centroid แกน x ตำแหน่งที่ 2
            y2 = point_2[1] # centroid แกน y ตำแหน่งที่ 2
            text_position = ( int((x1 + x2) / 2), int(((y1 + y2) / 2) + 15) ) # ตำแหน่งที่จะแสดงค่า distance
            # คำนวณระยะห่างระหว่างไม้บรรทัด (euclidean_distance)
            #ruler_pixels.append( math.sqrt( (x1 - x2)**2 + (y1 - y2)**2 ) )
            ruler_pixels[0] =  math.sqrt( (x1 - x2)**2 + (y1 - y2)**2 )
            #cv2.putText( img_for_show_distance, f"{round(ruler_pixels[0], 3)}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) )

        # มาร์กตำแหน่งของระยะไม้บรรทัด
        def mark_position( event, x, y, flags, param ) :
            # เมื่อคลิกซ้าย
            if event == cv2.EVENT_LBUTTONDOWN : 
                # คลิกได้ 2 ครั้ง
                if len( clicked_position ) < 2 :
                    clicked_position.append( (x, y) )
                    cv2.drawMarker( marker_img, (x, y), (0, 255, 0), markerType = cv2.MARKER_TILTED_CROSS, markerSize = 10, thickness = 2 ) # มาร์กจุดกลาง
                    cv2.imshow( "marker image", marker_img )
                    # หากมาร์กครบ 2 ตำแหน่งให้ลากเส้น
                    if len( clicked_position ) == 2 :
                        ruler_distance( marker_img, clicked_position[0], clicked_position[1] ) # คำนวณระยะห่างระหว่างจุดมาร์ก
                        cv2.line( marker_img, clicked_position[0], clicked_position[1], color = (0, 255, 0), thickness = 2 )
                        cv2.imshow( "marker image", marker_img )


        ruler_milimeter = 10 # กำหนดความยาวของไม้บรรทัดเป็น 20 mm.
        ruler_pixels = [0] # ความยาวของไม้บรรทัด เก็บเป็น pixels
        clicked_position = [] # เก็บตำแหน่ง x, y

        # นำเข้ารูปภาพ
        if select_cam == 1 : # มุมกล้อง 1
            #image = frame # รับเฟรมจากวิดีโอ
            marker_img = frame[self.region_1[1]:self.region_1[1]+self.region_1[3], self.region_1[0]:self.region_1[0]+self.region_1[2]]
        elif select_cam == 2 : # มุมกล้อง 2
            #image = frame # รับเฟรมจากวิดีโอ
            marker_img = frame[self.region_2[1]:self.region_2[1]+self.region_2[3], self.region_2[0]:self.region_2[0]+self.region_2[2]]
        elif select_cam == 3 : # มุมกล้อง 3
            #image = frame # รับเฟรมจากวิดีโอ
            marker_img = frame[self.region_3[1]:self.region_3[1]+self.region_3[3], self.region_3[0]:self.region_3[0]+self.region_3[2]]

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



    # เริ่มการทำงานของโปรแกรม
    def start_program( self, input_image : np.array ,ruler_mill_1, ruler_pix_1,ruler_mill_2, ruler_pix_2,ruler_mill_3, ruler_pix_3 ) :
        # สร้างหน้าต่างสำหรับ morphology operation
        kernel = np.ones( (9, 9), np.uint8 )

        # ตำแหน่งของหมุด
        green_position_1 = { 'x' : [0], 'y' : [0], 'w' : [0], 'h' : [0] }
        blue_position_1 = { 'x' : [0], 'y' : [0], 'w' : [0], 'h' : [0] }
        blue_position_2 = { 'x' : [0], 'y' : [0], 'w' : [0], 'h' : [0] }
        blue_position_3 = { 'x' : [0], 'y' : [0], 'w' : [0], 'h' : [0] }
        black_position_3 = { 'x' : [0], 'y' : [0], 'w' : [0], 'h' : [0] }

        # ตำแหน่งและค่าสำหรับลากเส้นระหว่างหมุด
        distance_green_1 = { 'x1' : 0, 'y1' : 0, 'x2' : 0, 'y2' : 0, 'distance' : 0, 'text position' : [0, 0] }
        distance_blue_1 = { 'x1' : 0, 'y1' : 0, 'x2' : 0, 'y2' : 0, 'distance' : 0, 'text position' : [0, 0] }
        distance_blue_2 = { 'x1' : 0, 'y1' : 0, 'x2' : 0, 'y2' : 0, 'distance' : 0, 'text position' : [0, 0] }
        distance_blue_3 = { 'x1' : 0, 'y1' : 0, 'x2' : 0, 'y2' : 0, 'distance' : 0, 'text position' : [0, 0] }
        distance_black_3 = { 'x1' : 0, 'y1' : 0, 'x2' : 0, 'y2' : 0, 'distance' : 0, 'text position' : [0, 0] }

        # สร้างภาพย่อยจากภาพหลัก
        # โดยระบุตำแหน่งจากอาร์เรย์ของภาพหลัก [ y : y + height, x : x + width ]
        sub_image_1 = input_image[self.pro_region_1[1]:self.pro_region_1[1]+self.pro_region_1[3], self.pro_region_1[0]:self.pro_region_1[0]+self.pro_region_1[2]]
        sub_image_2 = input_image[self.pro_region_2[1]:self.pro_region_2[1]+self.pro_region_2[3], self.pro_region_2[0]:self.pro_region_2[0]+self.pro_region_2[2]]
        sub_image_3 = input_image[self.pro_region_3[1]:self.pro_region_3[1]+self.pro_region_3[3], self.pro_region_3[0]:self.pro_region_3[0]+self.pro_region_3[2]]
    
        # แยกสีหมุดด้วยการใช้ภาพแบบ LAB และ CMYK
        # cam 1 (LAB)
        lab_img_1 = cv2.cvtColor( sub_image_1, cv2.COLOR_RGB2LAB )
        # cam 2 (LAB)
        lab_img_2 = cv2.cvtColor( sub_image_2, cv2.COLOR_RGB2LAB )
        # cam 3 (LAB, CMYK)
        lab_img_3 = cv2.cvtColor( sub_image_3, cv2.COLOR_RGB2LAB )
        cmyk_img_3 = self.bgr_2_cmyk( sub_image_3 )

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
        blur_green_img_1 = cv2.GaussianBlur( a_img_1, (9, 9 ), 10 )
        blur_blue_img_1 = cv2.GaussianBlur( b_img_1, (9, 9), 10 )
        # cam 2
        blur_blue_img_2 = cv2.GaussianBlur( b_img_2, (9, 9), 10 )
        # cam 3
        blur_blue_img_3 = cv2.GaussianBlur( b_img_3, (9, 9), 10 )
        blur_black_img_3 = cv2.GaussianBlur( k_img_3, (9, 9), 10 )
        # สำหรับมือและเวอร์เนียร์
        blur_semi_img_1 = cv2.GaussianBlur( a_img_1, (9, 9), 10 )
        blur_semi_img_2 = cv2.GaussianBlur( a_img_2, (9, 9), 10 )
        blur_semi_img_3 = cv2.GaussianBlur( a_img_3, (9, 9), 10 )

        # ทำการ thresholding
        # cam 1
        ret_a_1, thresh_green_img_1 = cv2.threshold( blur_green_img_1, self.green_threshold, 255, cv2.THRESH_BINARY_INV )
        ret_b_1, thresh_blue_img_1 = cv2.threshold( blur_blue_img_1, self.blue_threshold['cam 1'], 255, cv2.THRESH_BINARY_INV )
        # cam 2
        ret_b_2, thresh_blue_img_2 = cv2.threshold( blur_blue_img_2, self.blue_threshold['cam 2'], 255, cv2.THRESH_BINARY_INV )
        # cam 3
        ret_k_3, thresh_black_img_3 = cv2.threshold( blur_black_img_3, self.black_threshold, 255, cv2.THRESH_BINARY )
        ret_b_3, thresh_blue_img_3 = cv2.threshold( blur_blue_img_3, self.blue_threshold['cam 3'], 255, cv2.THRESH_BINARY_INV )
        # สำหรับมือและเวอร์เนียร์
        ret_semi, thresh_semi_img_1 = cv2.threshold( blur_semi_img_1, self.semi_threshold, 255, cv2.THRESH_BINARY_INV )
        ret_semi, thresh_semi_img_2 = cv2.threshold( blur_semi_img_2, self.semi_threshold, 255, cv2.THRESH_BINARY_INV )
        ret_semi, thresh_semi_img_3 = cv2.threshold( blur_semi_img_3, self.semi_threshold, 255, cv2.THRESH_BINARY_INV )

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
        detect = self.size_threshold( closing_green_img_1, 250, 600, 0, False ) # นับวัตถุที่เป็นสีขาว
        detect = self.size_threshold( closing_blue_img_1, 150, 400, 0, False ) # นับวัตถุที่เป็นสีขาว 80, 250
        # cam 2
        detect = self.size_threshold( closing_blue_img_2, 250, 550, 0, False )
        # cam 3
        detect = self.size_threshold( closing_black_img_3, 500, 1000, 0, False ) # นับวัตถุที่เป็นสีขาว 250, 600
        detect = self.size_threshold( closing_blue_img_3, 400, 1000, 0, False ) # นับวัตถุที่เป็นสีขาว 300, 700
        # มือ ไม้บรรทัด เวอร์เนียร์
        detect_1 = self.size_threshold( thresh_semi_img_1, 0, 0, 130000, True ) # 150000
        detect_2 = self.size_threshold( thresh_semi_img_2, 0, 0, 130000, True ) # 140000
        detect_3 = self.size_threshold( thresh_semi_img_3, 0, 0, 130000, True ) # 160000
        
        print( detect_1 )
        print( detect_2 )
        print( detect_3 )
        # ถ้าไม่มีมือและเวอร์เนียร์เข้ามาให้วัดหมุด
        if (detect_1 == True) and (detect_2 == True) and (detect_3 == True) :
            print( 'detect object' )
            # ตรวจความกลมของวัตถุ หากวัตถุไม่กลมให้ถมดำ
            # cam 1
            self.check_circularity( closing_green_img_1 )
            self.check_circularity( closing_blue_img_1 )
            # cam 2
            self.check_circularity( closing_blue_img_2 )
            # cam 3
            self.check_circularity( closing_black_img_3 )
            self.check_circularity( closing_blue_img_3 )
            
            # ตรวจจับหมุดและเหรียญ
            # cam 1
            green_position_1, centr_x_green_1, centr_y_green_1 = self.detect_object( sub_image_1, closing_green_img_1, 1 )
            blue_position_1, centr_x_blue_1, centr_y_blue_1 = self.detect_object( sub_image_1, closing_blue_img_1, 1 )
            # cam 2
            blue_position_2, centr_x_blue_2, centr_y_blue_2 = self.detect_object( sub_image_2, closing_blue_img_2, 2 )
            # cam 3
            black_position_3, centr_x_black_3, centr_y_black_3 = self.detect_object( sub_image_3, closing_black_img_3, 3 )
            blue_position_3, centr_x_blue_3, centr_y_blue_3 = self.detect_object( sub_image_3, closing_blue_img_3, 3 )

            # วัดระยะของหมุด
            # cam 1
            distance_green_1 = self.calculate_distance( sub_image_1, centr_x_green_1, centr_y_green_1, ruler_mill_1, ruler_pix_1 )
            distance_blue_1 = self.calculate_distance( sub_image_1, centr_x_blue_1, centr_y_blue_1, ruler_mill_1, ruler_pix_1 )
            # cam 2
            distance_blue_2 = self.calculate_distance( sub_image_2, centr_x_blue_2, centr_y_blue_2, ruler_mill_2, ruler_pix_2 )
            # cam 3
            distance_black_3 = self.calculate_distance( sub_image_3, centr_x_black_3, centr_y_black_3, ruler_mill_3, ruler_pix_3 )    
            distance_blue_3 = self.calculate_distance( sub_image_3, centr_x_blue_3, centr_y_blue_3, ruler_mill_3, ruler_pix_3 )


        # ค่าที่รีเทิร์นเป็น dictionary ที่เก็บค่า x, y, w, h
        return green_position_1, blue_position_1, blue_position_2, blue_position_3, black_position_3, distance_green_1, distance_blue_1, distance_blue_2, distance_blue_3, distance_black_3