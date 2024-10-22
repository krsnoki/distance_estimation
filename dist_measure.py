
import numpy as np
import cv2


#Define object specific variables  
dist = 0
focal = 1.7
pixels = 720
width_pixels = 4 * 1280


#find the distance from then camera
def get_dist(rectange_params,image):
    #find no of pixels covered
    pixels = rectange_params[1][0]
    print(pixels)
    #calculate distance
    dist = ((width_pixels*focal)/pixels)*100
    print("Distance: ", dist, "cm\n")
    #Wrtie n the image
    image = cv2.putText(image, 'Distance from Camera in CM :', org, font,  
       1, color, 2, cv2.LINE_AA)

    image = cv2.putText(image, str(dist), (110,50), font,  
       fontScale, color, 1, cv2.LINE_AA)

    return image

#Extract Frames 
cap = cv2.VideoCapture(2)


#basic constants for opencv Functs
kernel = np.ones((3,3),'uint8')
font = cv2.FONT_HERSHEY_SIMPLEX 
org = (0,20)  
fontScale = 0.6 
color = (0, 0, 255) 
thickness = 2


cv2.namedWindow('Object Dist Measure ',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object Dist Measure ', 700,600)


#loop to capture video frames
while True:
    ret, img = cap.read()

    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)


    # #predefined mask for green colour detection
    # lower = np.array([179, 61, 0])
    # upper = np.array([11, 33, 255])
    # mask = cv2.inRange(hsv_img, lower, upper)
     
    # Define the lower and upper bounds for red in HSV
    lower_red_1 = np.array([0, 70, 50])  # Lower bound for red hue
    upper_red_1 = np.array([10, 255, 255])  # Upper bound for red hue

    lower_red_2 = np.array([170, 70, 50])  # Lower bound for red hue (wrap-around)
    upper_red_2 = np.array([180, 255, 255])  # Upper bound for red hue (wrap-around)

    # Create the masks
    mask1 = cv2.inRange(hsv_img, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv_img, lower_red_2, upper_red_2)

    # Combine the masks to get the final mask for red
    mask_red = cv2.bitwise_or(mask1, mask2)


    d_img = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel,iterations = 5)


    #find the histogram
    cont, hei = cv2.findContours(d_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cont = sorted(cont, key = cv2.contourArea, reverse = True)[:1]

    for cnt in cont:
        #check for contour area
        if (cv2.contourArea(cnt)>100 and cv2.contourArea(cnt)<306000):

            #Draw a rectange on the contour
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect) 
            box = np.int0(box)
            cv2.drawContours(img,[box], -1,(255,0,0),3)
            
            img = get_dist(rect,img)

    cv2.imshow('Distance measured ',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()