import cv2
import numpy as np
from skimage import color


# Load image
main_image = cv2.imread( 'image_detection/frame500.jpg' )

# กำหนดตำแหน่งและขนาดของภาพย่อย
# ประกอบไปด้วย ( x, y, width, height )
region_1 = (481, 1, 719, 539)
region_2 = (1201, 1, 719, 539)
region_3 = (1201, 541, 719, 539)

# สร้างภาพย่อยจากภาพหลัก
# โดยระบุตำแหน่งจากอาร์เรย์ของภาพหลัก [ y : y + height, x : x + width ]
cam_1 = main_image[region_1[1]:region_1[1]+region_1[3], region_1[0]:region_1[0]+region_1[2]]
cam_2 = main_image[region_2[1]:region_2[1]+region_2[3], region_2[0]:region_2[0]+region_2[2]]
cam_3 = main_image[region_3[1]:region_3[1]+region_3[3], region_3[0]:region_3[0]+region_3[2]]

# แปลงจาก RGB เป็น YIQ
img_yiq = color.rgb2yiq( cam_3 )

ch_1, ch_2, ch_3 = cv2.split( img_yiq )

print(img_yiq.shape)

cv2.imshow( 'Channel 1', ch_1 )
cv2.imshow( 'Channel 2', ch_2 )
cv2.imshow( 'Channel 3', ch_3 )
cv2.imshow( 'YIQ image', img_yiq )
cv2.waitKey(0)
cv2.destroyAllWindows()