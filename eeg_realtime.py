import numpy as np
import pandas as pd
import sys
import json
import time
from telnetlib import Telnet


# raw eeg data received at a rate of 512 Hz
# summaries of eeg wave patterns provided by the interface at 1Hz intervals
# Initializing the arrays required to store the data.
attention_values = np.array([])
meditation_values = np.array([])
delta_values = np.array([])
theta_values = np.array([])
lowAlpha_values = np.array([])
highAlpha_values = np.array([])
lowBeta_values = np.array([])
highBeta_values = np.array([])
lowGamma_values = np.array([])
highGamma_values = np.array([])
blinkStrength_values = np.array([])
time_array = np.array([])

tn = Telnet('localhost', 13854)
start = time.perf_counter()

tn.write(str.encode('{"enableRawOutput": true , "format": "Json"}'))

outfile = "null"
if len(sys.argv) > 1:
    outfile = sys.argv[len(sys.argv) - 1]
    outfptr = open(outfile, 'w')

signalLevel = 0
eSenseDict = {'attention': 0, 'meditation': 0}
waveDict = {'lowGamma': 0, 'highGamma': 0, 'highAlpha': 0, 'delta': 0, 'highBeta': 0, 'lowAlpha': 0, 'lowBeta': 0,
            'theta': 0}


df = pd.DataFrame(columns=['delta', 'theta', 'lowAlpha',
                  'highAlpha', 'lowBeta', 'highBeta', 'lowGamma', 'highGamma'])

num = 0
id_name = "test"  # ID 입력

while time.perf_counter() - start < 93:

    num = num + 1
    blinkStrength = 0
    line = tn.read_until(b'\r')

    if len(line) > 20:
        timediff = time.perf_counter() - start
        dict = json.loads(str(line.decode('utf-8')))
        if "poorSignalLevel" in dict:
            signalLevel = dict['poorSignalLevel']
        if "blinkStrength" in dict:
            blinkStrength = dict['blinkStrength']
        if "eegPower" in dict:
            waveDict = dict['eegPower']
            eSenseDict = dict['eSense']

        outputstr = str(timediff) + ", " + str(signalLevel) + ", " + str(blinkStrength) + "," + str(
            eSenseDict['attention']) + ", " + str(eSenseDict['meditation']) + ", " + str(
            waveDict['lowGamma']) + ", " + str(waveDict['highGamma']) + ", " + str(waveDict['highAlpha']) + ", " + str(
            waveDict['delta']) + ", " + str(waveDict['highBeta']) + ", " + str(waveDict['lowAlpha']) + ", " + str(
            waveDict['lowBeta']) + ", " + str(waveDict['theta'])
        print(outputstr)

        if outfile != "null":
            outfptr.write(outputstr + "\n")

        Series = pd.DataFrame(
            {'delta': [waveDict['delta']],
             'theta': [waveDict['theta']], 'lowAlpha': [waveDict['lowAlpha']], 'highAlpha': [waveDict['highAlpha']],
             'lowBeta': [waveDict['lowBeta']], 'highBeta': [waveDict['highBeta']],
             'lowGamma': [waveDict['lowGamma']], 'highGamma': [waveDict['highGamma']]}, index=[str(int(timediff))])

        df = pd.concat([df, Series], axis=0)


filename = 'D:/exhibition/' + id_name + ".csv"


df.to_csv(filename)
tn.close()
