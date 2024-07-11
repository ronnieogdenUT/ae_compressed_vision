# This example sets up a balance controller using the Rotary Servo Disk.
# This example uses a Physical Rotary Servo (SRV-02),
# in a task-based (time-based IO) mode where you do not have to handle timing yourself.
# (task based mode is recommended for most applications).
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

# imports
from threading import Thread
import signal
import time
import math
import numpy as np
from pal.products.rot import RotaryServo
from pal.utilities.math import SignalGenerator, ddt_filter
from pal.utilities.scope import Scope

# Setup to enable killing the data generation thread using keyboard interrupts
global KILL_THREAD
KILL_THREAD = False

def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True

signal.signal(signal.SIGINT, sig_handler)

#region: Setup
simulationTime = 100 # will run for 30 seconds

scopePendulum = Scope(
    title='Pendulum encoder - alpha (rad)',
    timeWindow=10,
    xLabel='Time (s)',
    yLabel='Position (rad)')
scopePendulum.attachSignal(name='Pendulum - alpha (rad)',  width=1)

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

# Code to control the Rotary Inverted Pendulum
def control_loop():
    hardware = 1
    frequency = 500  # Hz, translates to sampling time 0.002s (same as MATLAB/Simulink)
    state_theta_dot = np.array([0, 0], dtype=np.float64)
    state_alpha_dot = np.array([0, 0], dtype=np.float64)

    # Limit sample rate for scope to 50 hz
    countMax = frequency / 50
    count = 0

    # controller gain K adopted from MATLAB/Simulink implementation
    K = np.array([-5.2612, 30.0353, -2.6407, 3.5405])

    with RotaryServo() as myRot:

        startTime = 0
        timeStamp = 0

        def elapsed_time():
            return time.time() - startTime

        startTime = time.time()

        while timeStamp < simulationTime and not KILL_THREAD:
            # Read sensor information
            myRot.read_outputs()

            theta = myRot.motorPosition * -1
            alpha_f =  myRot.pendulumPosition
            alpha = np.mod(alpha_f, 2*np.pi) - np.pi
            alpha_degrees = abs(math.degrees(alpha))

            # Calculate angular velocities with filter of 50 and 50 rad
            theta_dot, state_theta_dot = ddt_filter(theta, state_theta_dot, 50, 1/frequency)
            alpha_dot, state_alpha_dot = ddt_filter(alpha, state_alpha_dot, 50, 1/frequency)

            command_deg = 0

            states = command_deg * np.array([np.pi/180, 0, 0, 0]) - np.array([theta, alpha, theta_dot, alpha_dot])

            if alpha_degrees > 10 or alpha_degrees < -10:
                voltage = 0
            else:
                voltage = -1 * np.dot(K, states)

            # Write commands
            myRot.write_voltage(voltage)

            # Plot to scopes
            count += 1
            if count >= countMax:
                scopePendulum.sample(timeStamp, [states[1]])
                scopeBase.sample(timeStamp, [states[0]])
                scopeVoltage.sample(timeStamp, [voltage])
                count = 0

            timeStamp = elapsed_time()

# Setup data generation thread and run until complete
thread = Thread(target=control_loop)
thread.start()

while thread.is_alive() and (not KILL_THREAD):

    # This must be called regularly or the scope windows will freeze
    # Must be called in the main thread.
    Scope.refreshAll()
    time.sleep(0.01)

input('Press the enter key to exit.')