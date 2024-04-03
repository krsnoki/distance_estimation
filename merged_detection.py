import numpy as np
import cv2

# Object specific variables  
dist = 0
focal = 450
width = 4

# Define the function to calculate distance from camera
def get_dist(rectange_params, pixels, image):
    # Calculate distance
    dist = (width * focal) / pixels
    print("Distance: ", dist, "cm\n")
    
    # Write on the image
    org = (0, 20)
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = 0.6 
    color = (0, 0, 255) 
    image = cv2.putText(image, 'Distance from Camera in CM :', org, font, 1, color, 2, cv2.LINE_AA)
    image = cv2.putText(image, str(dist), (110, 50), font, fontScale, color, 1, cv2.LINE_AA)

    return image

# Function to resize images and concatenate them horizontally
def resize_final_img(x, y, *argv):
    images = cv2.resize(argv[0], (x, y))
    for i in argv[1:]:
        resize = cv2.resize(i, (x, y))
        images = np.concatenate((images, resize), axis=1)
    return images

# Trackbar callback function
def empty(a):
    pass

cap = cv2.VideoCapture(2)

# Create HSV trackbars
cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 300, 300)
cv2.createTrackbar("HUE Min", "HSV", 0, 179, empty)
cv2.createTrackbar("HUE Max", "HSV", 179, 179, empty)
cv2.createTrackbar("SAT Min", "HSV", 0, 255, empty)
cv2.createTrackbar("SAT Max", "HSV", 255, 255, empty)
cv2.createTrackbar("VALUE Min", "HSV", 0, 255, empty)
cv2.createTrackbar("VALUE Max", "HSV", 255, 255, empty)

# Basic constants for OpenCV functions
kernel = np.ones((3, 3), 'uint8')

cv2.namedWindow('Object Dist Measure', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object Dist Measure', 700, 600)

while True:
    ret, img = cap.read()
    if not ret:
        break

    h_min = cv2.getTrackbarPos("HUE Min", "HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv_img, lower, upper)

    # Remove extra garbage from image
    d_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=5)

    # Find the histogram
    cont, _ = cv2.findContours(d_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = sorted(cont, key=cv2.contourArea, reverse=True)[:1]

    for cnt in cont:
        # Check for contour area
        if 100 < cv2.contourArea(cnt) < 306000:
            # Draw a rectangle on the contour
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], -1, (255, 0, 0), 3)
            
            # Get distance and draw text
            pixels = rect[1][0]  # Width of the rectangle
            img = get_dist(rect, pixels, img)

    # Resize images and display
    final_img = resize_final_img(300, 300, mask,  d_img)
    cv2.imshow('Object Dist Measure', final_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
