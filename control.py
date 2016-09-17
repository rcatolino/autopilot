#!/usr/bin/python

import krpc
import math
import numpy
import time
from collections import deque
from matplotlib import pyplot
from matplotlib import animation
from numpy import fft


def bandstop(transform, bandwidth=3):
    ftransform = numpy.copy(transform)
    ftransform[0] = 0
    ftransform[1] = 0
    ftransform[2] = 0
    ftransform[-1] = 0
    return ftransform

class pid_controller:
    def __init__(self, setpoint_value, kp, ki, kd, kt, max_pitch=0):
        self.set_gains(kp, ki, kd)
        self.setpoint(setpoint_value)
        self.kt = kt
        self.max_pitch = max_pitch
        self.int_saturation = 0.5

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
        return (p + i + d, p, i, d)

class autopilot:
    def __init__(self, remote_address="128.0.0.1", resolution=100, pitch_target=0):
        self.client = krpc.connect(name="ap", address=remote_address)
        print("Connected to {}".format(self.client.krpc.get_status()))
        self.vessel = self.client.space_center.active_vessel
        print("Controling {}".format(self.vessel.name))

        self.position = self.vessel.flight(self.vessel.orbit.body.reference_frame)
        self.attitude = self.vessel.flight(self.vessel.surface_reference_frame)
        self.control = self.vessel.control

        self.time_interval = resolution # time between updates in ms
        self.pv_smoothed = 0
        self.pitch_target = pitch_target

        datapoint_nb = int(5*1000/self.time_interval) # holds 5*1000ms worth of data
        self.history = deque(maxlen=datapoint_nb)
        self.history.append(0)
        self.data_history = []
        for i in range(0, 5):
            self.data_history.append(deque(maxlen=datapoint_nb))
            self.data_history[i].append(0)

        self.pitch_controller = pid_controller(self.pitch_target/90, 3.5, 0.3, 0.01, self.time_interval/1000)
        self.roll_controller = pid_controller(0, 0.4, 0.1, 0.05, self.time_interval/1000)

    def update_callback(self, frame, axes, control_plot, process_plot, pv_freq_plot, pv_filtered_plot):
        attitude = self.attitude
        pitch = attitude.pitch
        print("pitch : {}\theading : {}\troll: {}\tAoA : {}".format(pitch, attitude.heading, attitude.roll, attitude.angle_of_attack))

        self.control.roll = self.roll_controller.control(attitude.roll/90)[0]
        if self.pv_smoothed != 0:
            print("last process pitch {}, corrected pitch {}".format(self.data_history[4][-1], self.pv_smoothed))
            process_pitch = self.pv_smoothed
        else:
            process_pitch = self.pitch_target

        (control_pitch, p, i, d) = self.pitch_controller.control(process_pitch/90)
        self.control.pitch = control_pitch

        pv_fft, pv_filtered, pv_noise = self.update_flight_data(pitch, control_pitch, p, i, d)
        return self.update_plots(axes, control_plot, process_plot, pv_freq_plot, pv_filtered_plot, pv_fft, pv_filtered, pv_noise)

    def update_flight_data(self, process_var, control_var, p, i, d):
        self.data_history[0].append(control_var)
        self.data_history[1].append(p)
        self.data_history[2].append(i)
        self.data_history[3].append(d)
        self.data_history[4].append(process_var)
        self.history.append(self.history[-1] + self.time_interval/1000)

        if len(self.history) > 6:
            pv_fft = fft.rfft(self.data_history[4])
            pv_filtered = bandstop(pv_fft)
            pv_noise = fft.irfft(pv_filtered, len(self.history))
            #self.pv_noise_correction = 2*pv_noise[-1] - pv_noise[-2] # crude estimation of the next value for the process variable noise
            self.pv_smoothed = self.data_history[4][-1] - pv_noise[-1] # process variable with estimated noise removed
        else:
            pv_fft = []
            pv_filtered = []
            pv_noise = []
        return pv_fft, pv_filtered, pv_noise

    def update_plots(self, control_axes, control, process, frequency, frequency_filtered, pv_fft, pv_filtered, pv_noise):
        process[0].set_data(self.history, self.data_history[4])
        control_axes.set_xlim(self.history[0], max(5, self.history[-1]))
        for i in range(0, len(self.data_history)-1):
            control[i].set_data(self.history, self.data_history[i])

        if len(pv_fft) > 0:
            freqs = fft.fftshift(fft.fftfreq(len(pv_fft)))
            frequency[0].set_data(freqs, numpy.abs(fft.fftshift(pv_fft)))
            frequency_filtered[0].set_data(freqs, numpy.abs(fft.fftshift(pv_filtered)))

            process[1].set_data(self.history, pv_noise)
            process[2].set_data(self.history, self.data_history[4] - pv_noise)

        return (control_axes, control, process, frequency, frequency_filtered)

    def run_ap(self):
        sas_state = self.control.sas
        self.control.sas = False
        fig = pyplot.figure()
        control_axes = fig.add_subplot(2,2,1)
        process_axes = fig.add_subplot(2,2,3, sharex=control_axes)
        pv_freq_axes = fig.add_subplot(2,2,2)
        pv_filtered_axes = fig.add_subplot(2,2,4, sharex=pv_freq_axes, sharey=pv_freq_axes)

        control_axes.set_xlim(0, 5)
        control_axes.set_ylim(-1.5, 1.5)
        process_axes.set_ylim(-40, 40)
        pv_freq_axes.set_xlim(-5, 5)
        pv_freq_axes.set_ylim(0, 80)

        control_plot = control_axes.plot([], [], 'b', [], [], 'c', [], [], 'r', [], [], 'm')
        process_plot = process_axes.plot([], [], 'b', [], [], 'r', [], [], 'm')
        pv_freq_plot = pv_freq_axes.plot([], [])
        pv_filtered_plot = pv_filtered_axes.plot([], [])
        control_axes.legend(control_plot, ['pitch control', 'p value', 'i value', 'd value'], loc=3)
        process_axes.legend(process_plot, ['pitch', 'pitch noise', 'filtered pitch'], loc=3)
        anim = animation.FuncAnimation(fig, self.update_callback, fargs=(control_axes, control_plot, process_plot, pv_freq_plot, pv_filtered_plot), interval=self.time_interval)
        try:
            pyplot.show()
        except KeyboardInterrupt:
            print("Leaving")
        finally:
            self.control.sas = sas_state


ap = autopilot(remote_address="192.168.2.3", pitch_target=4)
ap.run_ap()
