# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 19:08:25 2021

@author: Manuel Camargo
"""
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import getopt

from model_prediction import model_predictor as pr


# =============================================================================
# Main function
# =============================================================================
def catch_parameter(opt):
    """Change the captured parameters names"""
    switch = {'-h': 'help', '-a': 'activity', '-c': 'folder',
              '-b': 'model_file', '-v': 'variant', '-r': 'rep'}
    return switch.get(opt)


def main(argv):
    parameters = dict()
    column_names = {'Case ID': 'caseid',
                    'Activity': 'task',
                    'lifecycle:transition': 'event_type',
                    'Resource': 'user',
                    'time:timestamp': 'start_timestamp',
                    'time:timestamp': 'end_timestamp'}
    parameters['one_timestamp'] = True # Change this if only one timestamp in the log Else False
    parameters['is_single_exec'] = False
    parameters['read_options'] = {
        'timeformat': '%Y-%m-%d %H:%M:%S',
        'column_names': column_names,
        'one_timestamp': parameters['one_timestamp'],
        'filter_d_attrib': False}
    # Parameters settled manually or catched by console for batch operations
    if not argv:
        # predict_next, pred_sfx
        parameters['activity'] = 'pred_log'
        parameters['folder'] = '20241029_EB6917ED_690B_4C81_A70F_0643E29502E3'
        parameters['model_file'] = 'sepsis.h5'
        parameters['is_single_exec'] = True  # single or batch execution
        # variants and repetitions to be tested Random Choice, Arg Max
        parameters['variant'] = 'Random Choice' # use random choice sampling to select next activity
        parameters['rep'] = 10
    else:
        # Catch parms by console
        try:
            opts, _ = getopt.getopt(argv, "ho:a:f:c:b:v:r:",
                                    ['one_timestamp=', 'activity=', 'folder=',
                                     'model_file=', 'variant=', 'rep='])
            for opt, arg in opts:
                key = catch_parameter(opt)
                if key in ['rep']:
                    parameters[key] = int(arg)
                else:
                    parameters[key] = arg
        except getopt.GetoptError:
            print('Invalid option')
            sys.exit(2)
    print(parameters['folder'])
    print(parameters['model_file'])

    start_time = time.time()

    pr.ModelPredictor(parameters)

    end_time = time.time()

    print(f'Generation time (for {parameters["rep"]} log generations): {end_time - start_time} seconds.')


if __name__ == "__main__":
    main(sys.argv[1:])
