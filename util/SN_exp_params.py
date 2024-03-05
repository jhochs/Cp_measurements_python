# This file contains details of the Space Needle full-scale experiments
# Each record is a deployment and contains:
#  - file: where the Cpstats csv file is located
#  - motes: list of motes and where they were located on the building

# Correction for FS measurements based on LES results:
WDir_correction = 8
WS_correction = 0.76

FS = {
    0 : {
        'file' : '/Users/jackhochschild/Dropbox/School/Wind_Engineering/Sensor_Network/Code/Data/Seattle_deployment_2_May-July/Cpstats/SN_Cpstats_May-July_Gumbel_10min.csv',
        'motes' : {
            'CM17' : {'Degrees' : 348.5, 'Roof' : 'sloped'},
            'CM20' : {'Degrees' : 141, 'Roof' : 'flat'},
            'CM22' : {'Degrees' : 288.5, 'Roof' : 'sloped'},
            'CM35' : {'Degrees' : 48.5, 'Roof' : 'sloped'},
            'CM39' : {'Degrees' : 108.5, 'Roof' : 'sloped'}
        }
    },
    1 : {
        'file' : '/Users/jackhochschild/Dropbox/School/Wind_Engineering/Sensor_Network/Code/Data/Seattle_deployment_3_Nov-Dec/Cpstats/SN_Cpstats_Nov-Dec_Gumbel_10min.csv',
        'motes' : {
            'WM19' : {'Degrees' : 56, 'Roof' : 'flat'}
        }
    },
    2 : { 
        'file' : '/Users/jackhochschild/Dropbox/School/Wind_Engineering/Sensor_Network/Code/Data/Seattle_deployment_4_Dec-Jan/Cpstats/SN_Cpstats_Dec-Jan_Gumbel_10min.csv',
        'motes' : {
            'WM10' : {'Degrees' : 217, 'Roof' : 'flat'},
            'WM14' : {'Degrees' : 246, 'Roof' : 'flat'},
            'WM17' : {'Degrees' : 336, 'Roof' : 'flat'},
            'WM7' : {'Degrees' : 200, 'Roof' : 'sloped'},
            'WM18' : {'Degrees' : 230, 'Roof' : 'sloped'},
            'WM5' : {'Degrees' : 260, 'Roof' : 'sloped'},
            'WM23' : {'Degrees' : 320, 'Roof' : 'sloped'},
            'WM3' : {'Degrees' : 50, 'Roof' : 'sloped'}
        }
    },
    3 : {
        'file' : '/Users/jackhochschild/Dropbox/School/Wind_Engineering/Sensor_Network/Code/Data/Seattle_deployment_5_Feb/Cpstats/SN_Cpstats_Feb_Gumbel_10min.csv',
        'motes' : {
            'CM5' : {'Degrees' : 20, 'Roof' : 'sloped'},
            'WM4' : {'Degrees' : 95, 'Roof' : 'sloped'},
            'WM8' : {'Degrees' : 110, 'Roof' : 'sloped'},
            'WM12' : {'Degrees' : 190, 'Roof' : 'flat'},
            'WM14' : {'Degrees' : 246, 'Roof' : 'flat'},
            'WM18' : {'Degrees' : 230, 'Roof' : 'sloped'},
            'WM19' : {'Degrees' : 56, 'Roof' : 'flat'},
            'WM23' : {'Degrees' : 320, 'Roof' : 'sloped'},
            'WM25' : {'Degrees' : 290, 'Roof' : 'sloped'}
        }
    }
}