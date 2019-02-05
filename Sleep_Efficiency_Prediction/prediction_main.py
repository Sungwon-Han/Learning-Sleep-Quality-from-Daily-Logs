import imp
import csv
import json
import argparse
from data_util import *
from phase1_model import *
from phase2_model import *

def get_arguments(parser):
    parser.add_argument('--model_name', type=str, dest='model_name', help='Name of the model')
    parser.add_argument('--impute', type=str, dest='impute', help='Method for filling data, select from BLANK, AVERAGE, and GAIN')
    parser.add_argument('--load', action='store_true', help='With this option, load trained phase 1 and phase 2 model. Otherwise, train new one')
    parser.add_argument('--printlog', action='store_true', help='With this option, print train log in the prompt. Otherwise, no logs are printed')
    args = parser.parse_args()
    return args

def report_alphas(model_name, alphas, step_size):
    with open('result/attention_output_{}.csv'.format(model_name), 'w') as f:
        wr = csv.writer(f)
        user_count = 1
        day = 1
        wr.writerow(["User_Id", "Day", "Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Step 6", "Step 7"][:step_size + 2])
        for alpha_row in alphas:
            write_list = [user_count, day]
            if isinstance(alpha_row, np.float32):
                write_list.append(alpha_row)
            else:
                write_list.extend(alpha_row)
                
            wr.writerow(write_list)
            if day == 7:
                day = 1
                user_count += 1
            else:
                day += 1

def report_loss(model_name, loss_list):
    with open('result/loss_result_{}.csv'.format(model_name), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(["Phase1_step1", "Phase1_step2", "Phase1_step3", "Phase1_step4", "Phase1_step5", "Phase1_step6", "Phase1_step7", "Phase2"])
        wr.writerow(loss_list)


if __name__ == '__main__':
    # Load Arguments
    args = get_arguments(argparse.ArgumentParser())
    model_name = args.model_name
    printlog = args.printlog
    impute_method = args.impute
    load = args.load
        
    # Load Parameter
    with open('model_parameter.json') as f:
        parameter = json.load(f)
        phase1_parameter = parameter['phase1_parameter']
        phase2_parameter = parameter['phase2_parameter']    
    
    # Filling Missing Data
    if impute_method == "BLANK":
        impute_BLANK(save = True)
    elif impute_method == "AVERAGE":
        impute_AVERAGE(save = True)
    else:
        impute_GAIN()
    
    data, _, _ = load_dataset(impute_method)
    saved_train_index, saved_test_index = get_train_index(data)
    
    # Data Preprocess
    preprocessed_data, training, test = preprocess(data, saved_train_index, saved_test_index)
    
    # Phase 1
    loss_list = []
    training_phase2_h_list = []
    test_phase2_h_list = []
    block_hidden_list = []
    for step_size in range(1, 8):
        name = "Phase1_{}_{}".format(model_name, step_size)
        loss, alphas, y_hat, training_phase2_h, test_phase2_h = phase1_trainorload(name, training, test, step_size,
                                                                                   phase1_parameter['batch_size'], 
                                                                                   phase1_parameter['learning_rate'], 
                                                                                   phase1_parameter['step{}_hidden_size'.format(step_size)], 
                                                                                   phase1_parameter['epoch'], 
                                                                                   phase1_parameter['keep_prob'], 
                                                                                   printlog = printlog, 
                                                                                   load = load)
        loss_list.append(loss)
        block_hidden_list.append(phase1_parameter['step{}_hidden_size'.format(step_size)])
        training_phase2_h_list.append(training_phase2_h)
        test_phase2_h_list.append(test_phase2_h)
        
        if step_size > 1:
            report_alphas(name, alphas, step_size)

    # Phase 2
    metadata_train, metadata_test = get_metadata(np.array(preprocessed_data), loss_list, saved_train_index, saved_test_index)
    block_hidden_list.append(phase2_parameter['query_size'])
    name = "Phase2_{}".format(model_name)
    loss, alphas, y_hat = phase2_trainorload(name, 
                                             training_phase2_h_list, metadata_train, training[:, 7, 3], 
                                             test_phase2_h_list, metadata_test, test[:, 7, 3], 
                                             phase2_parameter['batch_size'], 
                                             phase2_parameter['learning_rate'], 
                                             block_hidden_list, 
                                             108, 
                                             [phase2_parameter['metadata_hidden_size1'], phase2_parameter['metadata_hidden_size2']], 
                                             phase2_parameter['epoch'], 
                                             phase2_parameter['keep_prob'],  
                                             printlog = printlog,
                                             load = load)

    loss_list.append(loss)
    report_alphas(name, alphas, 7)
    report_loss(model_name, loss_list)