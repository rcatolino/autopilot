#!/usr/bin/python

import krpc
import math
import numpy
import time
import sys
from collections import deque
from matplotlib import pyplot
from matplotlib import animation
from numpy import fft

def bandpass(transform, cutoff=0.1):
    ftransform = numpy.copy(transform)
    freqs = fft.fftfreq(len(ftransform))
    for i in range(0, len(ftransform)):
        if freqs[i] < -cutoff or freqs[i] > cutoff:
            ftransform[i] = 0
    return ftransform

def bandstop(transform, cutoff=0.1):
    ftransform = numpy.copy(transform)
    freqs = fft.fftfreq(len(ftransform))
    for i in range(0, len(ftransform)):
        if freqs[i] >= cutoff:
            break
        ftransform[i] = 0
    for i in range(len(ftransform)-1, 0, -1):
        if freqs[i] <= -cutoff:
            break
        ftransform[i] = 0
    return ftransform

class pid_controller:
    def __init__(self, setpoint_value, kp, ki, kd, time_resolution, time_window, max_pitch=0):
        self.set_gains(kp, ki, kd)
        self.setpoint(setpoint_value)
        self.kt = time_resolution/1000
        self.max_pitch = max_pitch
        self.int_saturation = 0.5

        history_size = int(time_window/time_resolution)
        self.p_history = deque(maxlen=history_size)
        self.i_history = deque(maxlen=history_size)
        self.d_history = deque(maxlen=history_size)
        self.cv_history = deque(maxlen=history_size)

    def setpoint(self, setpoint_value):
        self.sp = setpoint_value
        self.last_error = 0
        self.error_sum = 0

    def set_gains(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def control(self, process_var, pitch=0, log=False):
        pitch_corrector = 1
        if pitch != 0 and self.max_pitch != 0:
            pitch_corrector = 1 - (pitch/self.max_pitch)

        error = (self.sp - process_var) * pitch_corrector
        p = self.kp * error

        self.error_sum += error * self.kt
        if math.fabs(self.error_sum) * self.ki > self.int_saturation:
            self.error_sum = math.copysign(1, self.error_sum)*self.int_saturation/self.ki
        i = self.ki * self.error_sum

        d = self.kd * (error - self.last_error) / self.kt
        self.last_error = error

        if log:
            print("error {}, p {}, i {}, d {}".format(error, p, i, d))
        self.p_history.append(p)
        self.i_history.append(i)
        self.d_history.append(d)
        self.cv_history.append(p+i+d)
        return p + i + d

class freq_filter:
    def __init__(self, time_resolution, time_window, cutoff=0.1): # in ms
        self.cutoff = cutoff
        self.time_resolution = time_resolution
        history_size = int(time_window/time_resolution)
        self.raw_history = deque(maxlen=history_size)
        self.smoothed = []
        self.oscillation = []
        self.smoothed_history = deque(maxlen=history_size)
        self.oscillation_history = deque(maxlen=history_size)
        self.history = deque(maxlen=history_size)

    def update_data(self, process_var):
        try:
            self.history.append(self.history[-1] + self.time_resolution/1000)
        except IndexError:
            self.history.append(0)
        self.raw_history.append(process_var)
        if len(self.raw_history) > 6:
            transform = fft.rfft(self.raw_history)
            self.smoothed = fft.irfft(bandpass(transform, cutoff=self.cutoff), len(self.raw_history))
            self.smoothed_history.append(self.smoothed[-1])
            self.oscillation = fft.irfft(bandstop(transform, cutoff=self.cutoff), len(self.raw_history))
            self.oscillation_history.append(self.oscillation[-1])
        else:
            self.smoothed_history.append(process_var)
            self.oscillation_history.append(0)

    def plot_data(self, axe):
        plots = axe.get_lines()
        axe.set_xlim(self.history[0], max(1, self.history[-1]))

        if len(plots) == 0:
            for _ in range(0, 4):
                axe.plot([], [])
            plots = axe.get_lines()
            axe.legend(plots, ['raw history', 'instant oscillation', 'oscillation history', 'smoothed history'], loc=4)
            axe.set_ylim(-0.9, 0.9)

        plots[0].set_data(self.history, self.raw_history)
        if len(self.oscillation) > 0:
            plots[1].set_data(self.history, self.oscillation)
        plots[2].set_data(self.history, self.oscillation_history)
        plots[3].set_data(self.history, self.smoothed_history)

    def get_oscillation(self):
        return self.oscillation_history[-1]

    def get_smoothed(self):
        return self.smoothed_history[-1]

class autopilot:
    def __init__(self, remote_address="128.0.0.1", resolution=100, pitch_target=0):
        self.client = krpc.connect(name="ap", address=remote_address)
        print("Connected to {}".format(self.client.krpc.get_status()))
        self.vessel = self.client.space_center.active_vessel
        print("Controling {}".format(self.vessel.name))

        self.position = self.vessel.flight(self.vessel.orbit.body.reference_frame)
        self.attitude = self.vessel.flight(self.vessel.surface_reference_frame)
        self.control = self.vessel.control

        self.time_resolution = resolution # time between updates in ms
        self.time_window = 5000 # history range in ms
        self.pitch_target = pitch_target

        history_size = int(self.time_window/self.time_resolution) # holds 5000ms worth of data
        self.history = deque(maxlen=history_size)

        self.pitch_filter = freq_filter(self.time_resolution, self.time_window)
        self.pitch_noise = freq_filter(self.time_resolution, self.time_window/4, cutoff=0.05)
        self.roll_filter = freq_filter(self.time_resolution, self.time_window)
        self.roll_noise = freq_filter(self.time_resolution, self.time_window/5, cutoff=0.01)
        self.pitch_controller = pid_controller(self.pitch_target/90, 4, 0.5, 0, self.time_resolution, self.time_window)
        self.roll_controller = pid_controller(0, 0.3, 0, 0, self.time_resolution, self.time_window)

    def update_callback(self, frame, axes):
        attitude = self.attitude
        vessel_roll = attitude.roll
        self.roll_noise.update_data(vessel_roll/90)
        self.roll_filter.update_data(self.roll_controller.control(vessel_roll/90))
        self.control.roll = self.roll_filter.get_smoothed() - 0.5 * self.roll_noise.get_oscillation()

        vessel_pitch = attitude.pitch
        self.pitch_noise.update_data(vessel_pitch/90)
        self.pitch_filter.update_data(self.pitch_controller.control(vessel_pitch/90))
        self.control.pitch = self.pitch_filter.get_smoothed() - 0.8 * self.pitch_noise.get_oscillation()
        #print("process pitch {}, corrected pitch {}".format(control_pitch, control_pitch_corrected))

        try:
            self.history.append(self.history[-1] + self.time_resolution/1000)
        except IndexError:
            self.history.append(0)

        return self.update_plots(axes)

    def update_plots(self, axes):
        self.roll_noise.plot_data(axes[0])
        self.pitch_noise.plot_data(axes[1])
        return (axes,)

    def run_ap(self):
        sas_state = self.control.sas
        self.control.sas = False
        fig = pyplot.figure()

        axes = [fig.add_subplot(2,2,1), fig.add_subplot(2,2,2), fig.add_subplot(2,2,3), fig.add_subplot(2,2,4)]
        anim = animation.FuncAnimation(fig, self.update_callback, fargs=(axes,), interval=self.time_resolution)
        try:
            pyplot.show()
        except KeyboardInterrupt:
            print("Leaving")
        finally:
            self.control.sas = sas_state


ap = autopilot(remote_address="192.168.2.3", pitch_target=int(sys.argv[1]))
ap.run_ap()
