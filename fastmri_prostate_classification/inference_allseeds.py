import argparse
import numpy as np
import os
import torch
from sklearn import metrics
# from utils.load_fastmri_data_convnext_diff import load_data as load_data_diff
# from utils.load_fastmri_data_convnext_t2 import load_data as load_data_t2
from utils.custom_data_t2w_mask_adc import load_data
from model.model import ConvNext_model
import yaml
import matplotlib.pyplot as plt

def calculate_metrics(labels, binary_preds):

    # Calculate Precision, Recall, Accuracy, F1 Score, and Confusion Matrix
    precision = metrics.precision_score(labels, binary_preds)
    recall = metrics.recall_score(labels, binary_preds)
    accuracy = metrics.accuracy_score(labels, binary_preds)
    f1 = metrics.f1_score(labels, binary_preds)
    confusion_matrix = metrics.confusion_matrix(labels, binary_preds)

    return precision, recall, accuracy, f1, confusion_matrix

def test(model, test_loader, device):
    """
    Test the ConvNext model on the test set.

    Parameters:
    - model: The ConvNext model.
    - test_loader: The PyTorch DataLoader for the test set.
    - device: The device (CPU or GPU) on which to perform the testing.

    Returns:
    - auc_test (float): The area under the ROC curve on the test set.
    - raw_preds_test (Tensor): Concatenated raw predictions from the test set.
    """
    total_num_test, all_out, all_labels_test = 0,  [], [] 
    model.eval()                                                             
    with torch.no_grad():                                                    
        for _, (data, target) in enumerate(test_loader):
            data, target = data.to(device), torch.flatten(target.to(device))   
            out = model(data)                                                 

            out = torch.flatten(out)                                          
            
            all_out.append(out)                       
            all_labels_test.append(target)               
            total_num_test += 1                         

    all_labels_npy = torch.cat(all_labels_test).detach().cpu().numpy().astype(np.int32) 
    all_preds_npy = torch.sigmoid(torch.cat(all_out)).detach().cpu().numpy()           

    auc_test = metrics.roc_auc_score(all_labels_npy, all_preds_npy)                    
        
    return auc_test, all_preds_npy, all_labels_npy

def run_inference(config_t2):
    """
    Run an inference session on the trained ConvNext network

    Parameters:
    - config (dict): Configuration parameters for inference on the trained ConvNext model (same file used during training).
    """
    use_cuda = torch.cuda.is_available()                               
    device = torch.device("cuda" if use_cuda else "cpu")                
    print('Found this device:{}'.format(device))
    
    # _, _, test_loader_diff = load_data_diff(config_diff['data']['datasheet'],  config_diff["data"]["data_location"], int(config_diff['data']['norm_type']),  config_diff['training']['augment'], config_diff['training']['saveims'], config_diff['model_args']['rundir'])
    _, _, test_loader_t2 = load_data(
        config_t2,
        config_t2['data']['datapath'], 
        config_t2["data"]["labelpath"],
        config_t2["data"]["glandmask_path"], 
        int(config_t2['data']['norm_type']),  
        config_t2['training']['augment'], 
        config_t2['training']['saveims'], 
        config_t2['model_args']['rundir']
    )

    # _, _, test_loader_t2 = load_data_t2(config_t2['data']['datasheet'], config_t2["data"]["data_location"],  int(config_t2['data']['norm_type']),  config_t2['training']['augment'], config_t2['training']['saveims'], config_t2['model_args']['rundir'])

    # print('Lengths diffusion:Test:{}'.format(len(test_loader_diff)))  
    print('Lengths T2:Test:{}'.format(len(test_loader_t2)))  


    # model_diff = ConvNext_model(config_diff, diff = True)
    
    # model_path_diff = os.path.join(config_diff['model_args']['rundir'], "model_epoch_" + str(config_diff['load_model_epoch']) +'.pth')
    # print("Loading model:{}".format(model_path_diff))
    # model_diff.load_state_dict(torch.load(model_path_diff))
    # model_diff.to(device)

    model_t2 = ConvNext_model(config_t2)
    model_path_t2 = os.path.join(config_t2['model_args']['rundir'], 'best.pth')
    print("Loading model:{}".format(model_path_t2))
    model_t2.load_state_dict(torch.load(model_path_t2))
    model_t2.to(device)

    # Assuming `labels_t2` are the true labels and `raw_preds_test_t2` are the predicted probabilities:
    # Convert predicted probabilities to binary predictions (0 or 1) using a threshold (e.g., 0.5)

    # AUC_test_diff, raw_preds_test_diff, labels_diff  = test(model_diff, test_loader_diff, device)    
    AUC_test_t2, raw_preds_test_t2, labels_t2  = test(model_t2, test_loader_t2, device)    
    fpr_t2, tpr_t2, thresholds = metrics.roc_curve(labels_t2, raw_preds_test_t2)

    # Calculate Youden's J statistic
    j_scores = tpr_t2 - fpr_t2
    best_index = np.argmax(j_scores)
    youden_best_threshold = thresholds[best_index]

    # Calculate the Euclidean distance to (0,1) for each point on the ROC curve
    distances = np.sqrt((fpr)**2 + (1 - tpr)**2)
    optimal_idx = np.argmin(distances)
    optimal_threshold_euclidean = thresholds[optimal_idx]

    binary_preds_t2_youden = (raw_preds_test_t2 >= youden_best_threshold).astype(int)
    binary_preds_t2_min_euclidean = (raw_preds_test_t2 >= optimal_threshold_euclidean).astype(int)
    binary_preds_t2_50 = (raw_preds_test_t2 >= 0.5).astype(int)

    # Calculate Precision, Recall, Accuracy, F1 Score, and Confusion Matrix
    precision_t2, recall_t2, accuracy_t2, f1_t2, confusion_matrix_t2 = calculate_metrics(labels_t2, binary_preds_t2_youden)
    # Print the results
    print("Using Youden's threshold")
    print(f"Best threshold (Youden's J): {youden_best_threshold}")
    print(f"Test AUC - T2 is: {AUC_test_t2:.3f}")
    print(f"Precision - T2: {precision_t2:.3f}")
    print(f"Recall - T2: {recall_t2:.3f}")
    print(f"Accuracy - T2: {accuracy_t2:.3f}")
    print(f"F1 Score - T2: {f1_t2:.3f}")
    print(f"Confusion Matrix - T2:\n{confusion_matrix_t2}")

    precision_t2, recall_t2, accuracy_t2, f1_t2, confusion_matrix_t2 = calculate_metrics(labels_t2, binary_preds_t2_min_euclidean)
    # Print the results
    print("Using minimum euclidean distrance threshold")
    print(f"Best Threshold (Min Distance to (0,1)): {optimal_threshold_euclidean}")
    print(f"Test AUC - T2 is: {AUC_test_t2:.3f}")
    print(f"Precision - T2: {precision_t2:.3f}")
    print(f"Recall - T2: {recall_t2:.3f}")
    print(f"Accuracy - T2: {accuracy_t2:.3f}")
    print(f"F1 Score - T2: {f1_t2:.3f}")
    print(f"Confusion Matrix - T2:\n{confusion_matrix_t2}")

    precision_t2, recall_t2, accuracy_t2, f1_t2, confusion_matrix_t2 = calculate_metrics(labels_t2, binary_preds_t2_50)
    # Print the results
    print("Using 0.5 as threshold")
    print(f"Test AUC - T2 is: {AUC_test_t2:.3f}")
    print(f"Precision - T2: {precision_t2:.3f}")
    print(f"Recall - T2: {recall_t2:.3f}")
    print(f"Accuracy - T2: {accuracy_t2:.3f}")
    print(f"F1 Score - T2: {f1_t2:.3f}")
    print(f"Confusion Matrix - T2:\n{confusion_matrix_t2}")
    # print("Test AUC - diffusion is:{:.3f}".format(AUC_test_diff))
    
    # fpr_diff, tpr_diff, _ = metrics.roc_curve(labels_diff, raw_preds_test_diff)
    fpr_t2, tpr_t2, _ = metrics.roc_curve(labels_t2, binary_preds_t2_50)
    plt.title('Receiver Operating Characteristic')
    # plt.plot(fpr_diff, tpr_diff, 'b', label = 'AUC diff = %0.2f' % AUC_test_diff, c= 'red')
    plt.plot(fpr_t2, tpr_t2, 'b', label = 'AUC T2= %0.2f' % AUC_test_t2, c= 'blue')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1.02])
    plt.ylim([0, 1.02])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    save_png_file = os.path.join(config_t2['model_args']['rundir'], "Test_ROC_Curve_fastMRI_prostate.png")
    plt.savefig(save_png_file, bbox_inches = "tight")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    """
    Create an argument parser for the main script.

    Returns:
    - parser: The argparse.ArgumentParser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_t2', type=str, required=True)           # config file which has all the training inputs
    # parser.add_argument('--config_file_diff', type=str, required=True)         # config file which has all the training inputs
    parser.add_argument('--index_seed', type=int)                               # Optional: Seed number for reproducibility for all numpy, random, torch
    parser.add_argument('--concat_adc', type=str2bool, required=True, help='Set to True or False to specify whether to concatenate ADC as an additional channel.')
    parser.add_argument('--concat_mask', type=str2bool, required=True, help='Set to True or False to specify whether to concatenate gland mask as an additional channel.')
    parser.add_argument('--focal_loss', type=str2bool, default=False, help='whether to use focal loss instead of weighted bce')
    parser.add_argument('--use_2_5d', type=str2bool, required=True, help='Set to True or False to specify whether to use 2.5D.')

    return parser


if __name__ == '__main__':
    """
    Main script for training the ConvNext model.
    """
    args_con = get_parser().parse_args() 

    seed_list = [10383, 44820, 238, 3939, 74783, 92938, 143, 2992, 7373, 988]                                              
    
    # Check if a specific seed index is provided
    if args_con.index_seed is not None:
        # Use the specific seed from the list
        seed_select = seed_list[args_con.index_seed]
        seed_list = [seed_select]

    # Loop through all seeds
    for seed_select in seed_list:
        # Load config file
        with open(args_con.config_file_t2) as f:
            args_t2 = yaml.load(f, Loader=yaml.UnsafeLoader)  

        # with open(args_con.config_file_diff) as f:
        #     args_diff = yaml.load(f, Loader=yaml.UnsafeLoader)  

        args_t2['seed'] = seed_select
        args_t2['concat_mask'] = args_con.concat_mask
        args_t2['concat_adc'] = args_con.concat_adc
        args_t2['focal_loss'] = args_con.focal_loss
        args_t2['use_2_5d'] = args_con.use_2_5d
        print(args_t2)

        main_fol_t2 = args_t2["results_fol"]
        # main_fol_dwi = args_diff["results_fol"]
        subfolder = 't2w'  # Always include 't2w' as it's the base modality

        if args_t2['concat_adc']:
            subfolder += '_adc'
        if args_t2['concat_mask']:
            subfolder += '_mask'
        if args_t2['use_2_5d']:
            subfolder += '_use2_5d'

        args_t2['model_args']['rundir'] = os.path.join(main_fol_t2, subfolder, args_t2['model_args']['rundir'] + '_SEED_' + str(seed_select))
        print("Model rundir T2:{}".format(args_t2['model_args']['rundir']))
        # args_diff['model_args']['rundir'] = os.path.join(main_fol_dwi, args_diff['model_args']['rundir'] + '_SEED_' + str(seed_select)) 

        # print("Model rundir diffusion:{}".format(args_diff['model_args']['rundir']))   

        torch.manual_seed(seed_select)                                           
        torch.cuda.manual_seed(seed_select)                               
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed_select)

        run_inference(args_t2)

# %%