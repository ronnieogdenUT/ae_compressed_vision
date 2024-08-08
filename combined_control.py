# Code to control inverted pendulum first by sensor data, then event camera data.
# Requires file called StateEstimator containing the Pendulum class to exist on path.
# Adapted from balance_control_rip.py
# Pitch of pendulum is alpha
# Azimuth of pendulum is psi

# imports
import numpy as np
import cv2 as cv
import dv_processing as dv
import datetime
from threading import Thread
import signal
import time
import math
import matplotlib.pyplot as plt
from pal.products.rot import RotaryServo
from pal.utilities.math import ddt_filter
from pal.utilities.scope import Scope
from state_estimator import Pendulum

# Setup to enable killing the data generation thread using keyboard interrupts
global KILL_THREAD
KILL_THREAD = False

def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True

signal.signal(signal.SIGINT, sig_handler)

#region: Setup
simulationTime = 100 # will run for 100 seconds

scopePendulum = Scope(
    title='Pendulum encoder - alpha (rad)',
    timeWindow=10,
    xLabel='Time (s)',
    yLabel='Position (rad)')
scopePendulum.attachSignal(name='Pendulum - alpha (rad)', width=1)

scopeBase = Scope(
    title='Base encoder - theta (rad)',
    timeWindow=10,
    xLabel='Time (s)',
    yLabel='Position (rad)')
scopeBase.attachSignal(name='Base - theta (rad)',  width=1)

scopeVoltage = Scope(
    title='Motor Voltage',
    timeWindow=10,
    xLabel='Time (s)',
    yLabel='Voltage (volts)')
scopeVoltage.attachSignal(name='Voltage',  width=1)
#endregion

camera = dv.io.CameraCapture() 

# Initialize visualizer and filter instance 
resolution = camera.getEventResolution()
visualizer = dv.visualization.EventVisualizer(resolution)
filt = dv.noise.BackgroundActivityNoiseFilter(resolution, backgroundActivityDuration=datetime.timedelta(milliseconds=1))

# Calibration data
calib_mat = np.array([[4.7935785589484539e+02, 0., 1.7991454669785605e+02],[0., 4.8012492314777433e+02, 1.2720519077247445e+02],[0., 0., 1.]])

dist_coeff = np.array([-1.3575559636619938e+00, 1.6777048762210476e+00, -3.4979855013626326e-02, -1.2535521405803662e-02, -1.6079622172586507e+00])
                       
obj = Pendulum(camera, resolution, visualizer, filt, calib_mat, dist_coeff)
start_time = datetime.datetime.now()

# Plotting vars
sensor_pitch = [] # pitch data from sensor
sensor_azi = [] # azimuthal data from sensor
x_ax = [] # time
y_ax = [] # pitch from computer vision
z_ax = [] # azimuth from computer vision
weighted_azi = [] # slewed azimuthal measurements
weighted_pit = [] # slewed pitch measurements
sen_a_rad = [] # azimuthal data from sensor in radians
cam_a_rad = [] # azimuthal data from camera in radians

weight = 0

# values for moving averages
past_azi = []
azi_avg = 0
azi_avg_g = []
past_pit = []
pit_avg = 0
pit_avg_g = []
n = 30 # values to average over for azimuth
n_pit = 90 #values to average over for pitch

# Code to control the Rotary Inverted Pendulum
def control_loop():
    global weight, azi_avg, pit_avg
    hardware = 1
    frequency = 500  # Hz, translates to sampling time 0.002s
    state_theta_dot = np.array([0, 0], dtype=np.float64)
    state_alpha_dot = np.array([0, 0], dtype=np.float64)

    # Limit sample rate for scope to 50 Hz
    countMax = frequency / 50
    count = 0

    # Controller gain K adopted from MATLAB/Simulink implementation
    K = np.array([-5.2612, 30.0353, -2.6407, 3.5405]) # original from slack

    with RotaryServo() as myRot:
        startTime = 0
        timeStamp = 0 
            
        def elapsed_time():
            return time.time() - startTime

        startTime = time.time()

        while not KILL_THREAD:

            # Read sensor information
            myRot.read_outputs()

            # Azi filter
            if len(past_azi) < n and len(past_azi) > 0 and timeStamp > 17:
                azi_avg = sum(past_azi) / len(past_azi)
            elif len(past_azi) >= n and timeStamp > 17:
                azi_avg = sum(past_azi[-n:]) / n

            # Pitch filter-- not currently being used
            if len(past_pit) < n_pit and len(past_pit) > 0 and timeStamp > 17:
                pit_avg = sum(past_pit) / len(past_pit)
            elif len(past_pit) >= n_pit and timeStamp > 17:
                pit_avg = sum(past_pit[-n_pit:]) / n_pit

            if timeStamp < 17:
                # Setup time-- sensor data only
                theta = myRot.motorPosition 
                alpha_f = myRot.pendulumPosition
                alpha = np.mod(alpha_f, 2 * np.pi) - np.pi
                alpha_degrees = abs(math.degrees(alpha))
                weighted_azi.append(0)
                weighted_pit.append(0)
                azi_avg_g.append(0)
                pit_avg_g.append(0)
            elif weight <= 1:
                # Slewing from sensor to camera data
                # Azimuth weighting
                theta = ((1 - weight) * myRot.motorPosition) + (weight * azi_avg)
                weight = weight + 0.0001
                weighted_azi.append(theta)
                if obj.lines == True:
                    past_azi.append(obj.psi)
                azi_avg_g.append(azi_avg)

                # Pitch weighting-- not currently being used
                alpha_f = myRot.pendulumPosition
                alpha = np.mod(alpha_f, 2 * np.pi) - np.pi
                alpha_test = ((1 - weight) * alpha) + (weight * math.radians(pit_avg))
                alpha_degrees = abs(math.degrees(alpha))
                weight = weight + 0.0001
                weighted_pit.append(math.degrees(alpha_test))
                if obj.lines:
                    past_pit.append(obj.alpha_deg)
                pit_avg_g.append(pit_avg)
            else:
                # Entirely camera data
                theta = azi_avg
                weighted_azi.append(0)
                past_azi.append(obj.psi)
                azi_avg_g.append(azi_avg)

                # Pitch -- currently from sensor readings 
                alpha_f = myRot.pendulumPosition
                alpha = np.mod(alpha_f, 2 * np.pi) - np.pi
                alpha_degrees = pit_avg
                alpha_test= math.radians(alpha_degrees)
                weighted_pit.append(0)
                past_pit.append(obj.alpha_deg)
                pit_avg_g.append(pit_avg)

            # Calculate angular velocities with filter of 50 rad/s
            theta_dot, state_theta_dot = ddt_filter(theta, state_theta_dot, 50, 1 / frequency)
            alpha_dot, state_alpha_dot = ddt_filter(alpha, state_alpha_dot, 50, 1 / frequency)

            reference_state = np.zeros(4) # to zero
            states = reference_state * np.array([0, 0, 0, 0]) - np.array([theta, alpha, theta_dot, alpha_dot])

            if alpha_degrees > 10 or alpha_degrees < -10 :
                voltage = 0
            else:
                voltage = -1 * np.dot(K, states)

            # Write commands
            myRot.write_voltage(voltage)

            # Plot to scopes
            count += 1
            if count >= countMax:
                # Sensor Graphs
                scopePendulum.sample(timeStamp, [states[1]])
                scopeBase.sample(timeStamp, [states[0]])
                scopeVoltage.sample(timeStamp, [voltage])

                count = 0
                
            # Appending to plot
            timeStamp = elapsed_time()
            x_ax.append(timeStamp)
            alpha_f = myRot.pendulumPosition
            alpha_mod = np.mod(alpha_f, 2 * np.pi) - np.pi
            sensor_pitch.append(math.degrees(alpha_mod))
            sensor_azi.append(math.degrees(myRot.motorPosition))
            y_ax.append(obj.alpha_deg)
            z_ax.append(obj.psi_deg)
            sen_a_rad.append(myRot.motorPosition)
            cam_a_rad.append(obj.psi)

slicer = dv.EventStreamSlicer()
slicer.doEveryTimeInterval(datetime.timedelta(milliseconds=20), obj.getAngles)

# Start read loop
def read_events_loop():
    while not KILL_THREAD:
        events = camera.getNextEventBatch()

        if events is not None:
            filt.accept(events)
            filt_events = filt.generateEvents()
            slicer.accept(filt_events)

# Setup data generation thread and run until complete
control_thread = Thread(target=control_loop)
read_thread = Thread(target=read_events_loop)

control_thread.start()
read_thread.start()

while control_thread.is_alive() and not KILL_THREAD:
    # This must be called regularly or the scope windows will freeze
    Scope.refreshAll()
    time.sleep(0.01)

input('Press the enter key to exit.')
KILL_THREAD = True
control_thread.join()
read_thread.join()

cv.destroyAllWindows()

# Pitch camera plot
plt.figure(figsize=(10, 5))
plt.plot(x_ax, y_ax, label = "Camera")
plt.title('Pitch over Time')
plt.xlabel('Time (s)')
plt.ylabel('Pitch (Degrees)')
plt.plot(x_ax, sensor_pitch, label = "Sensor")
plt.plot(x_ax, weighted_pit, label = "Weighted")
plt.plot(x_ax, pit_avg_g, label = "Moving Average")
plt.legend()
plt.show()

# Azimuth plot
plt.figure(figsize = (10, 5))
plt.plot(x_ax, cam_a_rad, label = "Camera")
plt.title('Azimuth over Time')
plt.xlabel('Time (s)')
plt.ylabel('Azimuth (Rads)')
plt.plot(x_ax, sen_a_rad, label = "Sensor")
plt.plot(x_ax, weighted_azi, label = "Weighted")
plt.plot(x_ax, azi_avg_g, label = "Moving Average")
plt.legend()
plt.show()