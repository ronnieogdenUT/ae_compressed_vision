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
sen_a_rad = []
cam_a_rad = []

weight = 0

# values for moving averages
past_azi = []
azi_avg = 0
azi_avg_g = []
past_pit = []
pit_avg = 0
pit_avg_g = []
n = 17
n_pit = 90

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
    # limits: theta = 10 dgs / alpha = 5 dgs / theta_dot = 0.4 rads / alpha_dot = 0.3 rads / volt = 0.75
    K = np.array([-5.2612, 30.0353, -2.6407, 3.5405]) # original from slack
    #K = np.array([-1.09559128758809,	13.9289162172293,	-1.46333425098854,	1.96647670223037])
    #K = np.array([-1.54495816603853,	14.3203390065466,	-1.54431706920055,	2.04287793434126]) # theta = 8 dgs / theta_dot = 0.65 rads
    #K = ([-1.57480969149389,	14.4437069362606,	-1.55554430848033,	2.06382093511675]) # theta = 8 dgs / theta_dot = 0.65 rads / volt = 0.9
    #K = np.array([-1.88670644306782,  16.1111279690849,    -1.74700016025772,     2.27149056520451]) # Q for theta at 100
    K = np.array([-1.09559128758809,	11.5511299910944,	-1.19299588456263,	1.67239269529754]) 

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
                #azi_avg = obj.psi_deg
            elif len(past_azi) >= n and timeStamp > 17:
                azi_avg = sum(past_azi[-n:]) / n

            # Pitch filter
            if len(past_pit) < n_pit and len(past_pit) > 0 and timeStamp > 17:
                pit_avg = sum(past_pit) / len(past_pit)
                #azi_avg = obj.psi_deg
            elif len(past_pit) >= n_pit and timeStamp > 17:
                pit_avg = sum(past_pit[-n_pit:]) / n_pit

            if timeStamp < 17:
                theta = myRot.motorPosition 
                alpha_f = myRot.pendulumPosition
                alpha = np.mod(alpha_f, 2 * np.pi) - np.pi
                alpha_degrees = abs(math.degrees(alpha))
                weighted_azi.append(0)
                weighted_pit.append(0)
                azi_avg_g.append(0)
                pit_avg_g.append(0)
            elif weight <= 1:
                #theta = ((1 - weight) * myRot.motorPosition) + (weight * azi_avg))
                theta = ((1 - weight) * myRot.motorPosition) + (weight * obj.psi)
                weight = weight + 0.0001
                weighted_azi.append(theta)
                if obj.lines == True:
                    past_azi.append(obj.psi)
                azi_avg_g.append(azi_avg)

                # pitch weighting
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
                theta = azi_avg
                weighted_azi.append(0)
                past_azi.append(obj.psi)
                azi_avg_g.append(azi_avg)

                # pitch 
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
                # voltage = np.clip(voltage, -10, 10)
                # print(voltage)

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

            timeStamp = elapsed_time()
            x_ax.append(timeStamp)
            alpha_f = myRot.pendulumPosition
            alpha_mod = np.mod(alpha_f, 2 * np.pi) - np.pi
            sensor_pitch.append(math.degrees(alpha_mod))
            #sensor_azi.append(math.degrees(states[0]))
            sensor_azi.append(math.degrees(myRot.motorPosition))
            y_ax.append(obj.alpha_deg)
            z_ax.append(obj.psi_deg)
            sen_a_rad.append(myRot.motorPosition)
            cam_a_rad.append(obj.psi)
            #if obj.psi is not None:
                #cam_a_rad.append(obj.psi * -1)
            #else:
                #cam_a_rad.append(0)


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

# plt.figure(figsize = (10, 5))
# plt.plot(x_ax, z_ax, label = "Camera")
# plt.title('Azimuth over Time')
# plt.xlabel('Time (s)')
# plt.ylabel('Azimuth (Degrees)')
# plt.plot(x_ax, sensor_azi, label = "Sensor")
# plt.plot(x_ax, weighted_azi, label = "Weighted")
# plt.plot(x_ax, azi_avg_g, label = "Moving Average")
# plt.legend()
# plt.show()

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