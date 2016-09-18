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

class lowpass_filter:
    def __init__(self, time_resolution, time_window, history_window=0): # in ms
        if history_window == 0:
            history_window=time_window
        history_size = int(time_window/time_resolution)
        self.raw_history = deque(maxlen=history_size)
        self.smoothed = deque(maxlen=history_size)
        self.smoothed_history = deque(maxlen=int(history_window/time_resolution))
    def get_smoothed(self, process_var):
        self.raw_history.append(process_var)
        if len(self.raw_history) > 6:
            transform = fft.rfft(self.raw_history)
            self.smoothed = fft.irfft(bandpass(transform), len(self.raw_history))
            self.smoothed_history.append(self.smoothed[-1])
            return self.smoothed[-1]
        else:
            self.smoothed_history.append(process_var)
            return process_var

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

        self.pitch_filter = lowpass_filter(self.time_resolution, self.time_window)
        self.roll_filter = lowpass_filter(self.time_resolution, self.time_window/5, self.time_window)
        self.pitch_controller = pid_controller(self.pitch_target/90, 3.5, 0.3, 0.01, self.time_resolution, self.time_window)
        self.roll_controller = pid_controller(0, 0.3, 0.1, 0.01, self.time_resolution, self.time_window)

    def update_callback(self, frame, axes, subplots):
        attitude = self.attitude
        self.control.roll = self.roll_filter.get_smoothed(self.roll_controller.control(attitude.roll/90))
        control_pitch = self.pitch_controller.control(attitude.pitch/90)
        control_pitch_corrected = self.pitch_filter.get_smoothed(control_pitch)
        #print("process pitch {}, corrected pitch {}".format(control_pitch, control_pitch_corrected))
        self.control.pitch = control_pitch_corrected

        try:
            self.history.append(self.history[-1] + self.time_resolution/1000)
        except IndexError:
            self.history.append(0)

        return self.update_plots(axes, subplots)

    def update_plots(self, axes, subplots):
        pitch_plot = subplots[0]
        roll_plot = subplots[1]
        axes.set_xlim(self.history[0], max(5, self.history[-1]))

        pitch_plot[0].set_data(self.history, self.pitch_controller.p_history)
        pitch_plot[1].set_data(self.history, self.pitch_controller.i_history)
        pitch_plot[2].set_data(self.history, self.pitch_controller.d_history)
        pitch_plot[3].set_data(self.history, self.pitch_controller.cv_history)
        pitch_plot[4].set_data(self.history, self.pitch_filter.smoothed_history)

        roll_plot[0].set_data(self.history, self.roll_controller.p_history)
        roll_plot[1].set_data(self.history, self.roll_controller.i_history)
        roll_plot[2].set_data(self.history, self.roll_controller.d_history)
        roll_plot[3].set_data(self.history, self.roll_controller.cv_history)
        roll_plot[4].set_data(self.history, self.roll_filter.smoothed_history)

        """
        if len(pv_fft) > 0:
            freqs = fft.fftshift(fft.fftfreq(len(pv_fft)))
            frequency[0].set_data(freqs, numpy.abs(fft.fftshift(pv_fft)))
            frequency_filtered[0].set_data(freqs, numpy.abs(fft.fftshift(pv_filtered)))

            process[1].set_data(self.history, pv_smoothed)
        """

        return (axes, subplots)

    def run_ap(self):
        sas_state = self.control.sas
        self.control.sas = False
        fig = pyplot.figure()

        axes1 = fig.add_subplot(2,2,1)
        axes2 = fig.add_subplot(2,2,3, sharex=axes1, sharey=axes1)
        pv_freq_axes = fig.add_subplot(2,2,2)
        pv_filtered_axes = fig.add_subplot(2,2,4, sharex=pv_freq_axes, sharey=pv_freq_axes)

        axes1.set_xlim(0, 5)
        axes1.set_ylim(-0.5, 0.5)
        pv_freq_axes.set_xlim(-5, 5)
        pv_freq_axes.set_ylim(0, 80)

        plot1 = axes1.plot([], [], 'b', [], [], 'c', [], [], 'r', [], [], 'm', [], [], 'y')
        plot2 = axes2.plot([], [], 'b', [], [], 'c', [], [], 'r', [], [], 'm', [], [], 'y')
        pv_freq_plot = pv_freq_axes.plot([], [])
        pv_filtered_plot = pv_filtered_axes.plot([], [])
        axes1.legend(plot1, ['p value', 'i value', 'd value', 'pitch control', 'smoothed pitch control'], loc=3)
        axes2.legend(plot2, ['p value', 'i value', 'd value', 'roll control', 'smoothed roll control'], loc=3)
        anim = animation.FuncAnimation(fig, self.update_callback, fargs=(axes1, [plot1, plot2]), interval=self.time_resolution)
        try:
            pyplot.show()
        except KeyboardInterrupt:
            print("Leaving")
        finally:
            self.control.sas = sas_state


ap = autopilot(remote_address="192.168.2.3", pitch_target=int(sys.argv[1]))
ap.run_ap()
