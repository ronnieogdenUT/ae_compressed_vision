import datetime
import numpy as np
import dv_processing as dv
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# Open the camera
camera = dv.io.CameraCapture()

# Initialize visualizer and filter instance
resolution = camera.getEventResolution()
visualizer = dv.visualization.EventVisualizer(resolution)
filt = dv.noise.BackgroundActivityNoiseFilter(resolution, backgroundActivityDuration=datetime.timedelta(milliseconds=1))

# Calibration from calibration_camera_DAVIS346_00000807-2023_09_08_12_04_47
calib_mat = np.array([[4.7935785589484539e+02, 0., 1.7991454669785605e+02], [0., 4.8012492314777433e+02, 1.2720519077247445e+02], [0., 0., 1.]])
dist_coeff = np.array([-1.3575559636619938e+00, 1.6777048762210476e+00, -3.4979855013626326e-02, -1.2535521405803662e-02, -1.6079622172586507e+00])

# Create the window
cv.namedWindow("Event Slice Stream", cv.WINDOW_AUTOSIZE)

# Display parameters
thickness = 1
blue  = (255, 0, 0)
green = (0, 255, 0)
red   = (0, 0, 255)
black = (0, 0, 0)

# Initializing variables
x_ax = []  # Time stamps
y_ax = []  # Pitch data
z_ax = []  # Azimuth data
avg = 0
delta_x = None
alpha_old = None
psi_deg = None
start_time = datetime.datetime.now()

def get_lines(polarity, slice):
    image = None
    if polarity == True:
        filter = dv.EventPolarityFilter(polarity)
        filter.accept(slice)
        filtered_slice = filter.generateEvents()
        image = visualizer.generateImage(filtered_slice)
    else:
        filter = dv.EventPolarityFilter(polarity)
        filter.accept(slice)
        filtered_slice = filter.generateEvents()
        image = visualizer.generateImage(filtered_slice)

    # Undistort image
    h, w = image.shape[:2]
    new_cam_mat, roi = cv.getOptimalNewCameraMatrix(calib_mat, dist_coeff, (w,h), 1, (w,h))
    undist_image = cv.undistort(image, calib_mat, dist_coeff, None, new_cam_mat)
    # Crop the undistorted image
    x, y, w, h = roi
    image = undist_image[y+1:y+h, x:x+w]

    # Get lines
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    filtered_lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=105, maxLineGap=30)

    if filtered_lines is not None:
        filtered_lines = np.vstack(filtered_lines)
        return filtered_lines[0]
    else:
        print("No lines detected")
        return None
    
def sense_angles(event_slice):
    global delta_x, psi_deg, alpha_deg, x_ax, y_ax, z_ax, avg, current_time, alpha_old

    # Get image from visualizer
    image = visualizer.generateImage(event_slice)
    # Undistort image
    h, w = image.shape[:2]
    new_cam_mat, roi = cv.getOptimalNewCameraMatrix(calib_mat, dist_coeff, (w,h), 1, (w,h))
    undist_image = cv.undistort(image, calib_mat, dist_coeff, None, new_cam_mat)
    # Crop the undistorted image
    x, y, w, h = roi
    image = undist_image[y+1:y+h, x:x+w]

    # Get lines
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=105, maxLineGap=30) 

    if lines is not None:
        longest_lines = []
        for polarity in [True, False]:
            line = get_lines(polarity, event_slice)
            if line is not None:
                if line[1] < line[3]:
                    bot_y = line[3]
                    line[3] = line[1]
                    line[1] = bot_y
                    bot_x = line[2]
                    line[2] = line[0]
                    line[0] = bot_x
                longest_lines.append(line)
                pt1 = (int(line[0]), int(line[1]))
                pt2 = (int(line[2]), int(line[3]))
                cv.line(image, pt1, pt2, green, thickness)
        for line in lines:
            x1,y1,x2,y2=line[0]
            cv.line(image,(x1,y1),(x2,y2),green,thickness)
        x_top = 0
        y_top = 0
        x_bot = 0
        y_bot = 0
        if len(longest_lines) == 2:
            avg = np.mean(longest_lines, axis=0, dtype=np.int32)
            # Calculate coordinates
            pend_length = np.sqrt((avg[0] - avg[2])**2 + (avg[1] - avg[3])**2)
            if avg[3] > avg[1]:
                cv.line(image, (avg[2],(avg[3] - int(pend_length))), (avg[2],avg[3]), red, thickness)
                x_bot = avg[2]
                y_bot = avg[3]
                x_top = avg[0]
                y_top = avg[1]
            else:
                cv.line(image, (avg[0],(avg[1] - int(pend_length))), (avg[0],avg[1]), red, thickness)
                x_bot = avg[0]
                y_bot = avg[1]
                x_top = avg[2]
                y_top = avg[3]
        
            cv.line(image, (int(avg[0]), int(avg[1])), (int(avg[2]), int(avg[3])), red, thickness)
            # Find length of red line
            pend_length = np.sqrt((avg[0] - avg[2])**2 + (avg[1] - avg[3])**2)
        
            # Estimating pitch
            x_center = 80
            alpha = np.arctan2(y_bot - y_top, x_bot - x_top)
            alpha_deg = np.degrees(alpha)
            alpha_deg = 90 - alpha_deg
            cv.putText(image, str(alpha_deg), (10,30), cv.FONT_HERSHEY_COMPLEX , 0.3, green, thickness)

            # Estimating azimuth
            delta = 55 # Height from center of lens to bottom of pendulum
            r = 195
            x_prime = np.abs(x_bot - 80)
            y_prime = np.abs(y_bot - 75)
            psi = np.arcsin((x_prime * delta) / (y_prime * r))
            psi_deg = np.degrees(psi)
            psi_deg *= -1 if x_bot < x_center else 1
            cv.putText(image, str(psi_deg), (10, 60), cv.FONT_HERSHEY_COMPLEX, 0.3, blue, thickness)
            current_time = (datetime.datetime.now() - start_time).total_seconds()
            x_ax.append(current_time)
            y_ax.append(alpha_deg)
            z_ax.append(psi_deg)

            alpha_old = alpha_deg
    else:
        current_time = (datetime.datetime.now() - start_time).total_seconds()
        x_ax.append(current_time) 
        y_ax.append(alpha_old)
        z_ax.append(psi_deg)
    
    image = cv.resize(image, (0, 0), fx = 2.5, fy = 2.5)
    cv.imshow("Event Slice Stream", image)
    # Command to exit loop cleanly, destroys cv windows
    key = cv.waitKey(2)
    if key in [ord('q'), ord('Q')]:
        global looping
        looping = False

# Create an event slicer, this will only be used events only camera
slicer = dv.EventStreamSlicer()
slicer.doEveryTimeInterval(datetime.timedelta(milliseconds=25), sense_angles)

# start read loop
looping = True
while looping:
    # Get events
    events = camera.getNextEventBatch()
    # If no events arrived yet, continue reading
    if events is not None:
        filt.accept(events)
        filt_events = filt.generateEvents()
        slicer.accept(filt_events)
        
cv.destroyAllWindows()

plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(x_ax, y_ax)
plt.title('Pitch over Time')
plt.xlabel('Time (s)')
plt.ylabel('Pitch (Degrees)')
plt.subplot(2, 1, 2)
plt.plot(x_ax, z_ax)
plt.title('Azimuth over Time')
plt.xlabel('Time (s)')
plt.ylabel('Azimuth (Degrees)')
plt.tight_layout()
plt.show()