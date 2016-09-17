#!/usr/bin/python

import krpc
import math
import time
from collections import deque
from matplotlib import pyplot
from matplotlib import animation

class pid_controller:
    def __init__(self, setpoint_value, kp, ki, kd, kt, max_pitch=0):
        self.set_gains(kp, ki, kd)
        self.setpoint(setpoint_value)
        self.kt = kt
        self.max_pitch = max_pitch

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
        if math.fabs(self.error_sum) * self.ki > 0.4:
            self.error_sum = math.copysign(1, self.error_sum)*0.4/self.ki
        i = self.ki * self.error_sum

        d = self.kd * (error - self.last_error) / self.kt
        self.last_error = error

        if log:
            print("error {}, p {}, i {}, d {}".format(error, p, i, d))
        return p + i + d

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
        self.pitch_controller = pid_controller(3/90, 2, 0.2, 0.1, self.time_interval/1000)
        self.roll_controller = pid_controller(0, 0.4, 0.1, 0.05, self.time_interval/1000)
        #self.alt_controller = pid_controller(18000/25000, 3.5, 0.1, 0.55, 0.1, max_pitch=25)
        self.pitch_history = deque(maxlen=int(5*1000/self.time_interval)) # holds 5*1000ms worth of data
        self.history = deque(maxlen=int(5*1000/self.time_interval)) # holds 5*1000ms worth of data
        self.pitch_history.append(0)
        self.history.append(0)

    def plot_update(self, frame, axes, plot_data):
        #cr = pos_ctrl.vertical_speed
        #alt = p_flight.mean_altitude
        #print("climbing rate : {}\taltitude : {}".format(cr, alt))
        attitude = self.attitude
        print("pitch : {}\theading : {}\troll: {}\tAoA : {}".format(attitude.pitch, attitude.heading, attitude.roll, attitude.angle_of_attack))
        #c.pitch = alt_controller.control(p_flight.mean_altitude/25000, log=True, pitch=attitude.pitch)
        pitch = self.pitch_controller.control(attitude.pitch/90, log=True)
        self.pitch_history.append(pitch)
        self.history.append(self.history[-1] + self.time_interval/1000)
        self.control.pitch = pitch
        self.control.roll = self.roll_controller.control(attitude.roll/90)
        plot_data[0].set_data(self.history, self.pitch_history)
        axes.set_xlim(self.history[0], max(5, self.history[-1]))
        return (axes, plot_data)

    def run_ap(self):
        sas_state = self.control.sas
        self.control.sas = False
        fig = pyplot.figure()
        axes = pyplot.axes(ylim=(-1.5, 1.5), xlim=(0, 5))
        plot_data = axes.plot([], [])
        anim = animation.FuncAnimation(fig, self.plot_update, fargs=(axes, plot_data), interval=self.time_interval) # a frame every time_interval
        try:
            pyplot.show()
        except KeyboardInterrupt:
            print("Leaving")
        finally:
            self.control.sas = sas_state


game = ksp("192.168.2.3")
game.run_ap()
