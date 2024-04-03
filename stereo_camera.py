import numpy as np
import cv2

# Define object specific variables for both cameras
focal = 1.7
pixels = 720
width_pixels = 4 * 1280

# Function to calculate distance from camera
def get_dist(rectangle_params, image):
    # Find number of pixels covered
    pixels = rectangle_params[1][0]
    
    # Calculate distance in centimeters
    dist = ((width_pixels * focal) / pixels) * 100
    print("Distance: ", dist, "cm\n")
    
    # Write on the image
    org = (0, 20)
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = 0.6 
    color = (0, 0, 255) 
    image = cv2.putText(image, 'Distance from Camera in CM :', org, font, 1, color, 2, cv2.LINE_AA)
    image = cv2.putText(image, str(dist), (110, 50), font, fontScale, color, 1, cv2.LINE_AA)

    return image

# Create two video capture objects for each camera
cap1 = cv2.VideoCapture(2)
cap2 = cv2.VideoCapture(1)

# Basic constants for OpenCV functions
kernel = np.ones((3, 3), 'uint8')

while True:
    # Read frames from both cameras
    ret1, img1 = cap1.read()
    ret2, img2 = cap2.read()
    
    if not ret1 or not ret2:
        break
    
    # Convert frames to HSV for better color detection
    hsv_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for red in HSV
    lower_red_1 = np.array([0, 70, 50])  # Lower bound for red hue
    upper_red_1 = np.array([10, 255, 255])  # Upper bound for red hue

    lower_red_2 = np.array([170, 70, 50])  # Lower bound for red hue (wrap-around)
    upper_red_2 = np.array([180, 255, 255])  # Upper bound for red hue (wrap-around)

    # Create the masks for both cameras
    mask1 = cv2.inRange(hsv_img1, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv_img2, lower_red_2, upper_red_2)

    # Combine the masks to get the final mask for red
    mask_red1 = cv2.bitwise_or(mask1, mask2)
    mask_red2 = mask_red1.copy()  # Same mask for both cameras

    # Apply morphological operations to clean up the masks
    d_img1 = cv2.morphologyEx(mask_red1, cv2.MORPH_OPEN, kernel, iterations=5)
    d_img2 = cv2.morphologyEx(mask_red2, cv2.MORPH_OPEN, kernel, iterations=5)

    # Find the contours in both images
    cont1, _ = cv2.findContours(d_img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont2, _ = cv2.findContours(d_img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cont1 = sorted(cont1, key=cv2.contourArea, reverse=True)[:1]
    cont2 = sorted(cont2, key=cv2.contourArea, reverse=True)[:1]

    for cnt1, cnt2 in zip(cont1, cont2):
        # Check for contour area
        if 100 < cv2.contourArea(cnt1) < 306000 and 100 < cv2.contourArea(cnt2) < 306000:
            # Draw rectangles on the contours
            rect1 = cv2.minAreaRect(cnt1)
            box1 = cv2.boxPoints(rect1)
            box1 = np.int0(box1)
            cv2.drawContours(img1, [box1], -1, (255, 0, 0), 3)
            
            rect2 = cv2.minAreaRect(cnt2)
            box2 = cv2.boxPoints(rect2)
            box2 = np.int0(box2)
            cv2.drawContours(img2, [box2], -1, (255, 0, 0), 3)
            
            # Calculate distance for both cameras
            img1 = get_dist(rect1, img1)
            img2 = get_dist(rect2, img2)

    # Display both frames
    cv2.imshow('Camera 1', img1)
    cv2.imshow('Camera 2', img2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture objects and destroy windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
