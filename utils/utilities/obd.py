# Adapted from PYOBD
# Copyright 2004 Donour Sizemore (donour@uchicago.edu)
# Copyright 2009 Secons Ltd. (www.obdtester.com)
# Copyright 2016 Chuong Nguyen (chuong.nguyen@anu.edu.au)
# License GPL

import serial
import time
import string

from .obd_sensors import SENSORS
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)


class OBDPort(object):
    '''
    Interface to OBD II port. Adapt from PYOBD codes.
    Requires obd_sensors module from PYOBD.
    '''
    def __init__(self, portName='/dev/ttyUSB0', baudrate=38400, timeout=1,
                 noAttempt=3):
        self.port = serial.Serial(portName, baudrate, timeout=timeout)
        self.ELMver = "Unknown"
        self.State = 1 #state SERIAL is 1 connected, 0 disconnected (connection failed)
        count = 0
        while 1: #until error is returned try to connect
            try:
                self.send_command("atz")   # initialize
            except serial.SerialException:
                self.State = 0
                logging.error("Port {} is not available.".format(portName))
                return None

            self.ELMver = self.get_result()
            logging.info("atz response:" + self.ELMver)
            self.send_command("ate0")  # echo off
            logging.info("ate0 response:" + self.get_result())
            self.send_command("0100")
            ready = self.get_result()
            logging.info("0100 response1:" + ready)
            if ready == "SEARCHING...": #"BUSINIT: ...OK":
                ready=self.get_result()
                logging.info("0100 response2:" + ready)
                return None
            else:
                #ready=ready[-5:] #Expecting error message: BUSINIT:.ERROR (parse last 5 chars)
                logging.info("Connection attempt failed:" + ready)
                time.sleep(5)
                if count == noAttempt:
                    self.close()
                    self.State = 0
                    return None
                logging.info("Connection attempt:" + str(count))
                count = count+1

    def close(self):
        """ Resets device and closes all associated filehandles"""
        if (self.port is not None) and self.State == 1:
            self.port.close()
        self.port = None

    def send_command(self, cmd):
        """Internal use only: not a public interface"""
        if self.port:
            self.port.flushOutput()
            self.port.flushInput()
            for c in cmd:
                self.port.write(c)
            self.port.write("\r\n")
            logging.info("Send command:" + cmd)

    def get_result(self):
        """Internal use only: not a public interface"""
        time.sleep(0.1)
        if self.port:
            buffer = ""
            while 1:
                c = self.port.read(1)
                if c == '\r' and len(buffer) > 0:
                    break
                else:
                    if buffer != "" or c != ">": #if something is in buffer, add everything
                        buffer = buffer + c
            logging.info("Get result:" + buffer)
            return buffer
        else:
            logging.info("NO self.port!" + buffer)
        return None

    def interpret_result(self, code):
        """Internal use only: not a public interface"""
        # Code will be the string returned from the device.
        # It should look something like this:
        # '41 11 0 0\r\r'
        # 9 seems to be the length of the shortest valid response
        if len(code) < 7:
            raise "BogusCode"

        # get the first thing returned, echo should be off
        code = string.split(code, "\r")
        code = code[0]
        #remove whitespace
        code = string.split(code)
        code = string.join(code, "")
        #cables can behave differently
        if code[:6] == "NODATA": # there is no such sensor
            return "NODATA"

        # first 4 characters are code from ELM
        code = code[4:]
        return code

    # get sensor value from command
    def get_sensor_value(self, sensor):
        '''
        Input:
        - Sensor type
        Output:
         - Sensor value
         - Time stamp
         '''
        cmd = sensor.cmd
        self.send_command(cmd)
        data = self.get_result()
        tmsp = time.time()
        if data:
            data = self.interpret_result(data)
            if data != "NODATA":
                data = sensor.value(data)
        else:
            return "NORESPONSE"
        return data, tmsp

if __name__ == '__main__':
    port = OBDPort()
    for s in SENSORS:
        if s.name.strip() == 'Vehicle Speed': #'Coolant Temperature': #
            sensor = s

    save = True
    if save:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        fp = open('ODOMETRY_' + timestr + '.txt', 'w')
        fp.write('{}, {}, {}\n'.format('value', 'unit', 'timeStamp'))

    while True:
        try:
            value, tmsp = port.get_sensor_value(sensor)
            print('{} = {} {} at {}'.format(sensor.name, value, sensor.unit, tmsp))
            if save:
                fp.write('{}, {}, {}\n'.format(value, sensor.unit, tmsp))
        except:
            if save:
                fp.close()
            break
