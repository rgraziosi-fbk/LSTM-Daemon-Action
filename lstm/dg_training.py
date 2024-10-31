# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:27:58 2020

@author: Manuel Camargo
"""
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import getopt

from model_training import model_trainer as tr


# =============================================================================
# Main function
# =============================================================================
def catch_parameter(opt):
    """Change the captured parameters names"""
    switch = {'-h': 'help', '-f': 'file_name', '-m': 'model_family',
              '-e': 'max_eval', '-o': 'opt_method'}
    return switch.get(opt)


def main(argv):
    DATASET = 'sepsis'

    if DATASET == 'sepsis':
        timeformat = '%Y-%m-%d %H:%M:%S'
    elif DATASET in ['bpic2012_a', 'bpic2012_b']:
        timeformat = '%Y-%m-%d %H:%M:%S.%f'
    elif DATASET == 'traffic_fines':
        timeformat = '%Y-%m-%d'
    else:
        assert False

    parameters = dict()
    column_names = {'Case ID': 'caseid',
                    'Activity': 'task',
                    'lifecycle:transition': 'event_type',
                    'Resource': 'user',
                    'time:timestamp': 'start_timestamp',
                    'time:timestamp': 'end_timestamp'}
    parameters['one_timestamp'] = True  # Change this if only one timestamp in the log Else False
    parameters['read_options'] = {
        'timeformat': timeformat,
        'column_names': column_names,
        'one_timestamp': parameters['one_timestamp']}
    # Parameters settled manually or catched by console for batch operations
    if not argv:
        # Event-log filename
        parameters['file_name'] = f'{DATASET}.csv'
        parameters['model_family'] = 'lstm'
        parameters['opt_method'] = 'bayesian'  # 'rand_hpc', 'bayesian'
        parameters['max_eval'] = 10 # number of hyperparameter choices to try
    else:
        # Catch parms by console
        try:
            opts, _ = getopt.getopt(argv, "h:f:m:e:o:",
                                    ['file_name=', 'model_family=',
                                     'max_eval=', 'opt_method='])
            for opt, arg in opts:
                key = catch_parameter(opt)
                if key in ['max_eval']:
                    parameters[key] = int(arg)
                else:
                    parameters[key] = arg
        except getopt.GetoptError:
            print('Invalid option')
            sys.exit(2)
    # Similarity btw the resources profile execution (Song e.t. all)
    parameters['rp_sim'] = 0.85
    parameters['batch_size'] = 32  # Usually 32/64/128/256
    parameters['norm_method'] = ['max', 'lognorm']
    parameters['imp'] = 2
    parameters['epochs'] = 100
    parameters['n_size'] = [5, 10, 15,20,25,30,35,40,45,50]
    parameters['l_size'] = [50,100,150,200,250]
    parameters['lstm_act'] = ['selu', 'tanh','relu', 'sigmoid','linear','softmax']
    if parameters['model_family'] == 'lstm':
        parameters['model_type'] = ['specialized'] # to run other architectures change it to any of the following ['specialized','concatenated','shared_cat']

    parameters['dense_act'] = ['selu', 'tanh','relu', 'sigmoid','linear','softmax']
    parameters['optim'] = ['Nadam','Adam','SGD','Adagrad']

    if parameters['model_type'] == 'simple_gan':
        parameters['gan_pretrain'] = False
    parameters.pop('model_family', None)

    # Whether to filter or not by label ('deviant', 'regular' or None are possible values)
    parameters['filter_by_label'] = 'deviant'

    start_time = time.time()

    # Train models
    tr.ModelTrainer(parameters)

    end_time = time.time()

    print(f'Training time (for {parameters["max_eval"]} configurations, {parameters["epochs"]} epochs each): {end_time - start_time} seconds.')


if __name__ == "__main__":
    main(sys.argv[1:])
