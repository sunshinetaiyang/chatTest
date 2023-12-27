#mPythonType:0
from mpython import *

import time

def light_sensor():
    global x, y
    while True:
        if light.read() < 100:
            rgb.fill((int(255), int(255), int(255)))
            rgb.write()
            time.sleep_ms(1)
        else:
            rgb.fill( (0, 0, 0) )
            rgb.write()
            time.sleep_ms(1)

x = None

y = None

import math

def slip_ball():
    global x, y
    x = 64
    y = 32
    while True:
        oled.fill(0)
        oled.fill_circle(x, y, 15, 1)
        oled.show()
        if get_tilt_angle('X') >= 5:
            y = y + 1
        if get_tilt_angle('X') <= -5:
            y = y + -1
        if get_tilt_angle('Z') >= 5:
            x = x + 1
        if get_tilt_angle('Z') <= -5:
            x = x + -1

def get_tilt_angle(_axis):
    _Ax = accelerometer.get_x()
    _Ay = accelerometer.get_y()
    _Az = accelerometer.get_z()
    if 'X' == _axis:
        _T = math.sqrt(_Ay ** 2 + _Az ** 2)
        if _Az < 0: return math.degrees(math.atan2(_Ax , _T))
        else: return 180 - math.degrees(math.atan2(_Ax , _T))
    elif 'Y' == _axis:
        _T = math.sqrt(_Ax ** 2 + _Az ** 2)
        if _Az < 0: return math.degrees(math.atan2(_Ay , _T))
        else: return 180 - math.degrees(math.atan2(_Ay , _T))
    elif 'Z' == _axis:
        _T = math.sqrt(_Ax ** 2 + _Ay ** 2)
        if (_Ax + _Ay) < 0: return 180 - math.degrees(math.atan2(_T , _Az))
        else: return math.degrees(math.atan2(_T , _Az)) - 180
    return 0
light_sensor()
