# Chuong Nguyen 2016
# Water detection project
#
# Class to read GPS data
# Author: Chuong Nguyen <chuong.nguyen@anu.edu.au>
#
# License: BSD 3 clause

import serial
import time
import string
import io
import os
import sys
import struct
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)


class GPSPort(object):
    '''
    Interface to serial port from 3DR GPS/Compass module via Pixhawk4
    '''
    def __init__(self, portName='/dev/ttyUSB0', baudrate=115200, timeout=None,
                 noAttempt=50, gps_header=0x48234567):
        self.port = serial.Serial(portName, baudrate, timeout=timeout)
        self.State = 0
        self.header = gps_header
        self.firstByte = chr(self.header/0xffffff)
        self.noAttempt = noAttempt
        self.readGPS()  # try reading data
        self.State = 1
        logging.info('Port is opened.')

    def close(self):
        if (self.port is not None) and self.State == 1:
            self.port.close()
        self.port = None

    def readGPS(self):
        '''
        Read GPS data. Data is send in a group of 4 int32 numbers and 7 float32 numbers:
        - The first number is header
        - The next 3 numbers are longitude, latitude and altitude.
          Longitude and latitude need to be divivided by 1e7 to get degree value
        - The next 4 numbers are quaternion of orientation
        - The next 3 numbers are rollrate, pitchrate, yawrate
        - The last is the sum of all these numbers, except the header
        Output:
        - longitude, latitude and altitude, orientation quaternion, rollrate, pitchrate, yawrate
        - Time stamp
        '''
        for i in range(self.noAttempt):
            #print('Attemp %d' % i)
            # search for first byte of header
            x = self.port.read()
            if x == self.firstByte:
                #time.sleep(0.01)
                # read the remaing 3 bytes + bytes of three int32 numbers
                s = self.port.read(12*4-1)
                s2 = x+s  # concantenate the two parts
                tmsp = time.time()

                # convert to int32 numbers
                nums = []
                for j in range(4):
                    num = struct.unpack('>i', s2[j*4:j*4+4])[0]
                    nums.append(num)
                for j in range(4, 12):
                    num = struct.unpack('>f', s2[j*4:j*4+4])[0]
                    nums.append(num)
                # check header and average to make sure the data is valid
                # avg = (lon+lat+alt + q0+q1+q2+q3 + rollrate+pitchrate+yawrate) / 10
                sum_all = 0.0
                for num in nums[1:11]:
                    sum_all += num
                avg = sum_all//10 + 18 # need to add 18, don't know why
                if nums[0] == self.header: # and avg == nums[-1]:
                    lon = nums[1]/1e7
                    lat = nums[2]/1e7
                    alt = nums[3]
                    q = nums[4:8]
                    print(q)
                    rollrate = nums[8]
                    pitchrate = nums[9]
                    yawrate = nums[10]
                    return lon, lat, alt, q, rollrate, pitchrate, yawrate, tmsp
                logging.info('Garbage data')

        raise serial.SerialException

if __name__ == '__main__':
    port = GPSPort()

    save = True
    if save:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        fp = open('GPS_' + timestr + '.txt', 'w')
        fp.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format('latitude', 'longitude', 'altitude', 'quaternion','rollrate', 'pitchrate', 'yawrate', 'timeStamp'))

    while True:
        try:
            lon, lat, alt, q, rollrate, pitchrate, yawrate, tmsp = port.readGPS()
            #print(lon, lat, alt, q, rollrate, pitchrate, yawrate, tmsp)
            #print('lat, lon, alt, tmsp = %f, %f, %f, %d' % (lat, lon, alt, tmsp))
            if save:
                #fp.write('{}, {}, {}, {}\n'.format(lat, lon, alt, tmsp))
                fp.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(lon, lat, alt, q, rollrate, pitchrate, yawrate, tmsp))
        except:
            if save:
                fp.close()
            break
