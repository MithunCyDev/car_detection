import cv2
import numpy as np

# Create a VideoCapture object
video_cap = cv2.VideoCapture("car.mp4")

# Initialize the background subtractor object
background_sub = cv2.createBackgroundSubtractorMOG2()

# Define the kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Loop through the video frames
while True:
    # Read the video frame
    success, frame = video_cap.read()
  
    # If there are no more frames to show, break the loop
    if not success:
        break
    
    # Apply the background subtractor to the frame
    mask = background_sub.apply(frame)
    
    # Perform morphological opening operation on the mask
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Get the foreground frame using the mask and the original frame
    new_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Get the contours of the moving objects
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop over the contours
    for contour in contours:
        # If the contour area is too small, ignore it
        if cv2.contourArea(contour) > 1000:
            # Get the bounding rectangle of the contour
            (x, y, w, h) = cv2.boundingRect(contour)
            # Draw the bounding rectangle on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Frame", frame)
    
    # If the 'q' key is pressed, stop the loop
    if cv2.waitKey(30) == ord("q"): 
        break
    
# Release the video capture object
video_cap.release()
cv2.destroyAllWindows()
