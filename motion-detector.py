import cv2, time

# A variable that will save the first image as a background
first_frame = None

# Create video object
video = cv2.VideoCapture(0)


while True:

    # A numpy array that represents the image and boolean variable to check if it's proper
    check , frame = video.read()

    # Converts the image to be grayscale and gaussianBlur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21), 0)

    # Set the background for the first image
    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame , gray)

    # Apply thresh hold
    thresh_frame = cv2.threshold(delta_frame , 30 , 255 , cv2.THRESH_BINARY)[1]

    # Expand the contours
    thresh_frame = cv2.dilate(thresh_frame , None , iterations = 2)

    # Find the contours, saving them and checking their location
    (cnts,_) = cv2.findContours(thresh_frame.copy() , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    
    # Save the contours of objects of significant size
    for  contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue

    # If the size of the object is significant then we will circle it in a rectangle

    # Show the image captured on camera
    cv2.imshow("Capturing", gray)
    cv2.imshow("Delta Frame" , delta_frame)
    cv2.imshow("thresh_frame" , thresh_frame)



    key = cv2.waitKey(1)
    print(gray)

    # Pressing the letter 'q' will stop the loop
    if key == ord('q'):
        break


# Make sure the camera is released
video.release()
cv2.destroyAllWindows()

