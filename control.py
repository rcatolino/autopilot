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
    ftransform = transform
    ftransform[0] = 0
    ftransform[1] = 0
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

class ksp:
    def __init__(self, game_address="128.0.0.1", resolution=100):
        self.client = krpc.connect(name="ap", address=game_address)
        print("Connected to {}".format(self.client.krpc.get_status()))
        self.vessel = self.client.space_center.active_vessel
        print("Controling {}".format(self.vessel.name))
        self.position = self.vessel.flight(self.vessel.orbit.body.reference_frame)
        self.attitude = self.vessel.flight(self.vessel.surface_reference_frame)
        self.control = self.vessel.control
        self.time_interval = resolution # time between updates in ms
        self.pitch_controller = pid_controller(6/90, 3.5, 0.3, 0.01, self.time_interval/1000)
        self.roll_controller = pid_controller(0, 0.4, 0.1, 0.05, self.time_interval/1000)
        #self.alt_controller = pid_controller(18000/25000, 3.5, 0.1, 0.55, 0.1, max_pitch=25)
        self.data_history = []
        self.pv_correction = 0
        datapoint_nb = int(5*1000/self.time_interval) # holds 5*1000ms worth of data
        for i in range(0, 5):
            self.data_history.append(deque(maxlen=datapoint_nb))
            self.data_history[i].append(0)

        self.history = deque(maxlen=datapoint_nb)
        self.history.append(0)

    def plot_update(self, frame, axes, control_plot, parameter_plot, pv_transform_plot, cv_transform_plot):
        #cr = pos_ctrl.vertical_speed
        #alt = p_flight.mean_altitude
        #print("climbing rate : {}\taltitude : {}".format(cr, alt))
        attitude = self.attitude
        apitch = attitude.pitch
        #print("pitch : {}\theading : {}\troll: {}\tAoA : {}".format(apitch, attitude.heading, attitude.roll, attitude.angle_of_attack))
        #c.pitch = alt_controller.control(p_flight.mean_altitude/25000, log=True, pitch=attitude.pitch)
        self.control.roll = self.roll_controller.control(attitude.roll/90)[0]
        print("correction {}, pitch {}, pitch corrected {}".format(self.pv_correction, apitch, apitch - self.pv_correction))
        (pitch, p, i, d) = self.pitch_controller.control((apitch - self.pv_correction)/90)
        self.control.pitch = pitch

        self.data_history[0].append(pitch)
        self.data_history[1].append(p)
        self.data_history[2].append(i)
        self.data_history[3].append(d)
        self.data_history[4].append(apitch)
        self.history.append(self.history[-1] + self.time_interval/1000)

        for i in range(0, len(self.data_history)-1):
            control_plot[i].set_data(self.history, self.data_history[i])

        parameter_plot[0].set_data(self.history, self.data_history[4])
        axes.set_xlim(self.history[0], max(5, self.history[-1]))

        if len(self.history) > 10:
            pv_transform = fft.rfft(self.data_history[4])
            cv_transform = fft.rfft(self.data_history[0])
            freqs = fft.fftshift(fft.fftfreq(len(cv_transform)))
            pv_transform_plot[0].set_data(freqs, numpy.abs(fft.fftshift(pv_transform)))
            cv_transform_plot[0].set_data(freqs, numpy.abs(fft.fftshift(bandstop(pv_transform))))

            ipv_transform = fft.irfft(bandstop(pv_transform), len(self.history))
            parameter_plot[1].set_data(list(self.history)[0:len(ipv_transform)], ipv_transform)
            parameter_plot[2].set_data(self.history, self.data_history[4] - ipv_transform)
            print("last itransform {}, last pitch {}, diff {}".format(ipv_transform[-1], apitch, apitch - ipv_transform[-1]))
            self.pv_correction = 2*ipv_transform[-1] - ipv_transform[-2]

        return (axes, control_plot, parameter_plot, pv_transform_plot, cv_transform_plot)

    def run_ap(self):
        sas_state = self.control.sas
        self.control.sas = False
        fig = pyplot.figure()
        control_axes = fig.add_subplot(2,2,1)
        parameter_axes = fig.add_subplot(2,2,3, sharex=control_axes)
        pv_transform_axes = fig.add_subplot(2,2,2)
        cv_transform_axes = fig.add_subplot(2,2,4, sharex=pv_transform_axes)

        control_axes.set_xlim(0, 5)
        control_axes.set_ylim(-1.5, 1.5)
        parameter_axes.set_ylim(-40, 40)
        pv_transform_axes.set_xlim(-5, 5)
        pv_transform_axes.set_ylim(0, 80)
        cv_transform_axes.set_ylim(0, 50)

        control_plot = control_axes.plot([], [], 'b', [], [], 'c', [], [], 'r', [], [], 'm')
        parameter_plot = parameter_axes.plot([], [], 'b', [], [], 'r', [], [], 'm')
        pv_transform_plot = pv_transform_axes.plot([], [])
        cv_transform_plot = cv_transform_axes.plot([], [])
        control_axes.legend(control_plot, ['pitch input', 'p value', 'i value', 'd value'], loc=3)
        parameter_axes.legend(parameter_plot, ['pitch', 'pitch noise', 'filtered pitch'], loc=3)
        anim = animation.FuncAnimation(fig, self.plot_update, fargs=(control_axes, control_plot, parameter_plot, pv_transform_plot, cv_transform_plot), interval=self.time_interval) # a frame every time_interval
        try:
            pyplot.show()
        except KeyboardInterrupt:
            print("Leaving")
        finally:
            self.control.sas = sas_state


game = ksp("192.168.2.3")
game.run_ap()
