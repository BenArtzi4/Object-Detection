import cv2, time, pandas
from datetime import datetime

# A variable that will save the first image as a background
first_frame = None

# An array that will contain the times when a new object is observed
times = []
status_list = [None,None]
df = pandas.DataFrame(columns = ["Start" , "End"])


# Create video object
video = cv2.VideoCapture(0 , cv2.CAP_DSHOW)

time.sleep(2)


while True:

    # A numpy array that represents the image and boolean variable to check if it's proper
    check , frame = video.read()
    status = 0

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
    (cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Save the contours of large objects
    for  contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue

        # Large object observed
        status = 1

        # If the object is large then we will create a rectangle the size of the object and then add it to the frame 
        (x , y , w , h ) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x , y), (x+w , y+h), (0,255,0), 3)

    status_list.append(status)

    # Adds the time a new object observed and dissapeared 
    status_list.append(status)
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())


    # Show the image captured on camera
    cv2.imshow("Capturing", gray)
    cv2.imshow("Delta Frame" , delta_frame)
    cv2.imshow("Thresh_frame" , thresh_frame)
    cv2.imshow("Color Frame" , frame)

    key = cv2.waitKey(1)

    # Pressing the letter 'q' will stop the loop
    if key == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break

end = None

if len(times) % 2 != 0:
    end = len(times)-1
else:
    end = len(times)


# Adds the information about the object that appeared and disappeared to the data frame
for i in range (0 , end , 2):
    df = df.append({"Start": times[i],"End": times[i+1]} , ignore_index = True)


print(times)
# Exports the data as a file
df.to_csv("Times.csv")

# Make sure the camera is released
video.release()
cv2.destroyAllWindows()

