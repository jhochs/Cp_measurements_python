import numpy as np

# This file contains details of the Space Needle full-scale experiments
# Each record is a deployment and contains:
#  - file: where the Cpstats csv file is located
#  - motes: list of motes and where they were located on the building 
#           (measured in centimeters, clockwise from NW corner)

# Correction for FS measurements based on LES results:
WDir_correction = 0
WS_correction = 1.83

edge_length = 3718  # length of each side of 650 California, in cm

FS = {
    0 : {
        'file' : '/Users/jackhochschild/Dropbox/School/Wind_Engineering/Sensor_Network/Code/Data/650Cal_deployment_1_Apr-May/Cpstats/650Cal_Cpstats_Gumbel_10min.csv',
        'motes' : {
            'CM32' : {'Position' : 1020, 'Type' : 'onboard'},
            'CM12' : {'Position' : edge_length-1060, 'Type' : 'onboard'},
            'CM19' : {'Position' : edge_length+280, 'Type' : 'onboard'},
            'CM14' : {'Position' : edge_length+1780, 'Type' : 'onboard'},
            'CM37' : {'Position' : 2*edge_length-220, 'Type' : 'onboard'},
            'CM33' : {'Position' : 2*edge_length+140, 'Type' : 'onboard'},
            'CM10' : {'Position' : 2*edge_length+1000, 'Type' : 'onboard'},
            'CM11' : {'Position' : 2*edge_length+1920, 'Type' : 'onboard'},
            'CM5'  : {'Position' : 3*edge_length-933, 'Type' : 'onboard'},
            'CM45' : {'Position' : 3*edge_length-120, 'Type' : 'onboard'},
            'CM17' : {'Position' : 3*edge_length+1070, 'Type' : 'onboard'},
            'CM39' : {'Position' : 4*edge_length-1160, 'Type' : 'onboard'},
            'CM20' : {'Position' : [1104, 784, np.nan], 'Type' : 'tethered'},
            'CM27' : {'Position' : [edge_length-1076, edge_length-884, np.nan], 'Type' : 'tethered'},
            'CM4'  : {'Position' : [edge_length+173, edge_length+108, np.nan], 'Type' : 'tethered'},  # swapped from on diagram
            'CM34' : {'Position' : [edge_length+1815, np.nan, np.nan], 'Type' : 'tethered'},
            'CM30' : {'Position' : [2*edge_length-312, 2*edge_length-150, np.nan], 'Type' : 'tethered'},
            'CM42' : {'Position' : [2*edge_length+136, 2*edge_length+240, np.nan], 'Type' : 'tethered'},
            'CM44' : {'Position' : [2*edge_length+890, 2*edge_length+950, np.nan], 'Type' : 'tethered'},
            'CM18' : {'Position' : [2*edge_length+1694, 2*edge_length+1981, np.nan], 'Type' : 'tethered'},
            'CM25' : {'Position' : [3*edge_length-938, 3*edge_length-858, np.nan], 'Type' : 'tethered'},
            'CM36' : {'Position' : [3*edge_length-228, 3*edge_length-168, 3*edge_length-91], 'Type' : 'tethered'},
            'CM26' : {'Position' : [3*edge_length+207, np.nan, np.nan], 'Type' : 'tethered'},
            'CM22' : {'Position' : [3*edge_length+1085, 3*edge_length+1007, np.nan], 'Type' : 'tethered'},
            'CM38' : {'Position' : [4*edge_length-1028, 4*edge_length-1160, np.nan], 'Type' : 'tethered'}
        }
    },
    1 : {
        'file' : '/Users/jackhochschild/Dropbox/School/Wind_Engineering/Sensor_Network/Code/Data/650Cal_deployment_2_May-Jun/Cpstats/650Cal_Cpstats_Gumbel_10min.csv',
        'motes' : {
            'CM27' : {'Position' : [462, 272, np.nan], 'Type' : 'tethered'},
            'CM4'  : {'Position' : [edge_length-1977, edge_length-1348, np.nan], 'Type' : 'tethered'}, # swapped from on diagram
            'CM32' : {'Position' : [edge_length-481, edge_length-228, np.nan], 'Type' : 'tethered'},
            'CM42' : {'Position' : [2*edge_length+557, 2*edge_length+427, np.nan], 'Type' : 'tethered'},
            'CM44' : {'Position' : [2*edge_length+1375, 2*edge_length+1469, np.nan], 'Type' : 'tethered'},
            'CM18' : {'Position' : [3*edge_length-1546, 3*edge_length-1332, np.nan], 'Type' : 'tethered'},
            'CM17' : {'Position' : [3*edge_length-640, 3*edge_length-573, np.nan], 'Type' : 'tethered'},
            'CM36' : {'Position' : [3*edge_length-405, 3*edge_length-335, 3*edge_length-277], 'Type' : 'tethered'},
            'CM11' : {'Position' : [3*edge_length+207, np.nan, np.nan], 'Type' : 'tethered'},
            'CM22' : {'Position' : [3*edge_length+786, 3*edge_length+697, np.nan], 'Type' : 'tethered'},
            'CM38' : {'Position' : [np.nan, 3*edge_length+1491, np.nan], 'Type' : 'tethered'}, # 1st sensor was : 3*edge_length+1616
            'CM20' : {'Position' : [4*edge_length-391, 4*edge_length-615, np.nan], 'Type' : 'tethered'}
        }
    }
}