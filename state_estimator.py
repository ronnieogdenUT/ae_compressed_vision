import numpy as np
import cv2 as cv
import dv_processing as dv
import datetime

class Pendulum:
    def __init__(self, cam, res, vis, filt, calib, dist):

        # Open the camera
        self.camera = cam

        self.start_time = datetime.datetime.now()
        self.x_ax = []  # Time stamps
        self.y_ax = []  # Pitch data
        self.z_ax = []  # Azimuth data

        # Initialize visualizer and filter instance 
        self.resolution = res
        self.visualizer = vis
        self.filt = filt

        # Calibration from calibration_camera_DAVIS346_00000807-2023_09_08_12_04_47
        self.calib_mat = calib
        self.dist_coeff = dist

        self.alpha_old = None
        self.psi_deg = None
        self.alpha_deg = None
        self.p_deg_g = None
        self.psi = None

        self.pitch = None
        self.azimuth = None
        self.lines = True

    def get_lines(self, polarity, slice):
    
        image = None

        # filtering image by polarity
        if polarity == True:
            filter = dv.EventPolarityFilter(polarity)
            filter.accept(slice)
            filtered_slice = filter.generateEvents()
            image = self.visualizer.generateImage(filtered_slice)
        else:
            filter = dv.EventPolarityFilter(polarity)
            filter.accept(slice)
            filtered_slice = filter.generateEvents()
            image = self.visualizer.generateImage(filtered_slice)
        
        h, w = image.shape[:2]
        new_cam_mat, roi = cv.getOptimalNewCameraMatrix(self.calib_mat, self.dist_coeff, (w,h), 1, (w,h))
        undist_image = cv.undistort(image, self.calib_mat, self.dist_coeff, None, new_cam_mat)
        # Crop the undistorted image
        x, y, w, h = roi
        image = undist_image[y+1:y+h, x:x+w]
            
        # Get lines
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 50, 150, apertureSize=3)
        filtered_lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=105, maxLineGap=30)

        # return line from image
        if filtered_lines is not None:
            filtered_lines = np.vstack(filtered_lines)
            line_lengths = ((filtered_lines[:, 2] - filtered_lines[:, 0])**2 + (filtered_lines[:, 3] - filtered_lines[:, 1])**2)**0.5
            longest_index = np.argmax(line_lengths)
            return filtered_lines[0]
    
    
    def getAngles(self, event_slice):
        global delta_x, psi_deg, alpha_deg, n, x_ax, y_ax, z_ax, avg, current_time, alpha_old
        # Get image from visualizer
        image = self.visualizer.generateImage(event_slice)
    
        # Undistort image
        h, w = image.shape[:2]
        new_cam_mat, roi = cv.getOptimalNewCameraMatrix(self.calib_mat, self.dist_coeff, (w,h), 1, (w,h))
        undist_image = cv.undistort(image, self.calib_mat, self.dist_coeff, None, new_cam_mat)
        # Crop the undistorted image
        x, y, w, h = roi
        image = undist_image[y+1:y+h, x:x+w]
        
    
        # Get lines
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 50, 150, apertureSize=3)
        lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=105, maxLineGap=30)

        # get blue and black lines after filtering by polarity
        if lines is not None:
            longest_lines = []
            for polarity in [True, False]:
                line = self.get_lines(polarity, event_slice)
                if line is not None:
                    if line[1] < line[3] :
                        bot_y = line[3]
                        line[3] = line[1]
                        line[1] = bot_y
                        bot_x = line[2]
                        line[2] = line[0]
                        line[0] = bot_x
                    longest_lines.append(line) 
            x_top = 0 
            y_top = 0
            x_bot = 0
            y_bot = 0

            self.lines = True
            
            # do if both blue and black line present
            if len(longest_lines) == 2:
                avg = np.mean(longest_lines, axis=0, dtype=np.int32)
        
                # get end coordinates of lines
                if avg[3] > avg[1]:
                    x_bot = avg[2]
                    y_bot = avg[3]
                    x_top = avg[0]
                    y_top = avg[1]
                else:
                    x_bot = avg[0]
                    y_bot = avg[1]
                    x_top = avg[2]
                    y_top = avg[3]
            
                # Estimate pitch angle
                alpha = np.arctan2(y_bot - y_top, x_bot - x_top)
                self.alpha_deg = np.degrees(alpha)
                self.alpha_deg = (90 - self.alpha_deg) # *-1
                
                # Estimate azimuth angle
                x_center = 80
                delta = 60 # 55 mm
                r = 190 # 195 mm
                x_prime = np.abs(x_bot - 80)
                y_prime = np.abs(y_bot - 75)
                self.psi = (np.arcsin((x_prime * delta) / (y_prime * r))) * -1
                self.psi_deg = np.degrees(self.psi)

                self.psi_deg *= -1 if x_bot < x_center else 1
                self.psi *= -1 if x_bot > x_center else 1

                current_time = (datetime.datetime.now() - self.start_time).total_seconds()
                self.z_ax.append(self.psi_deg)

                self.alpha_old = self.alpha_deg

        else:
            # run if no lines appear
            current_time = (datetime.datetime.now() - self.start_time).total_seconds()
            self.z_ax.append(self.psi_deg)
            self.lines = False