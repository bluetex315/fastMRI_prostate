import argparse
import numpy as np
import os
import torch
from sklearn import metrics
from utils.custom_data_adc_mask_t2w import load_data
from model.model import ConvNext_model
import yaml
import matplotlib.pyplot as plt

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

def run_inference(config_dwi):
    """
    Run an inference session on the trained ConvNext network

    Parameters:
    - config (dict): Configuration parameters for inference on the trained ConvNext model (same file used during training).
    """
    use_cuda = torch.cuda.is_available()                               
    device = torch.device("cuda" if use_cuda else "cpu")                
    print('Found this device:{}'.format(device))
    
    # _, _, test_loader_diff = load_data_diff(config_diff['data']['datasheet'],  config_diff["data"]["data_location"], int(config_diff['data']['norm_type']),  config_diff['training']['augment'], config_diff['training']['saveims'], config_diff['model_args']['rundir'])
    _, _, test_loader_dwi = load_data(
        config_dwi,
        config_dwi['data']['datapath'], 
        config_dwi["data"]["labelpath"],
        config_dwi["data"]["glandmask_path"], 
        int(config_dwi['data']['norm_type']),  
        config_dwi['training']['augment'], 
        config_dwi['training']['saveims'], 
        config_dwi['model_args']['rundir']
    )

    # _, _, test_loader_t2 = load_data_t2(config_t2['data']['datasheet'], config_t2["data"]["data_location"],  int(config_t2['data']['norm_type']),  config_t2['training']['augment'], config_t2['training']['saveims'], config_t2['model_args']['rundir'])

    # print('Lengths diffusion:Test:{}'.format(len(test_loader_diff)))  
    print('Lengths DWI: Test:{}'.format(len(test_loader_dwi)))  


    # model_diff = ConvNext_model(config_diff, diff = True)
    
    # model_path_diff = os.path.join(config_diff['model_args']['rundir'], "model_epoch_" + str(config_diff['load_model_epoch']) +'.pth')
    # print("Loading model:{}".format(model_path_diff))
    # model_diff.load_state_dict(torch.load(model_path_diff))
    # model_diff.to(device)

    model_dwi = ConvNext_model(config_dwi, diff=True)
    model_path_dwi = os.path.join(config_dwi['model_args']['rundir'], 'best.pth')
    print("Loading model:{}".format(model_path_dwi))
    model_dwi.load_state_dict(torch.load(model_path_dwi))
    model_dwi.to(device)

    # Assuming `labels_t2` are the true labels and `raw_preds_test_t2` are the predicted probabilities:
    # Convert predicted probabilities to binary predictions (0 or 1) using a threshold (e.g., 0.5)

    # AUC_test_diff, raw_preds_test_diff, labels_diff  = test(model_diff, test_loader_diff, device)    
    AUC_test_dwi, raw_preds_test_dwi, labels_dwi = test(model_dwi, test_loader_dwi, device)    
    fpr_dwi, tpr_dwi, thresholds = metrics.roc_curve(labels_dwi, raw_preds_test_dwi)

    # Calculate Youden's J statistic
    j_scores = tpr_dwi - fpr_dwi
    best_index = np.argmax(j_scores)
    best_threshold = thresholds[best_index]

    print(f"Best threshold (Youden's J): {best_threshold}")

    binary_preds_dwi = (raw_preds_test_dwi >= 0.5).astype(int)

    # Calculate Precision, Recall, Accuracy, F1 Score, and Confusion Matrix
    precision_dwi = metrics.precision_score(labels_dwi, binary_preds_dwi)
    recall_dwi = metrics.recall_score(labels_dwi, binary_preds_dwi)
    accuracy_dwi = metrics.accuracy_score(labels_dwi, binary_preds_dwi)
    f1_dwi = metrics.f1_score(labels_dwi, binary_preds_dwi)
    confusion_matrix_dwi = metrics.confusion_matrix(labels_dwi, binary_preds_dwi)

    # Print the results
    print(f"Test AUC - DWI is: {AUC_test_dwi:.3f}")
    print(f"Precision - DWI: {precision_dwi:.3f}")
    print(f"Recall - DWI: {recall_dwi:.3f}")
    print(f"Accuracy - DWI: {accuracy_dwi:.3f}")
    print(f"F1 Score - DWI {f1_dwi:.3f}")
    print(f"Confusion Matrix - DWI:\n{confusion_matrix_dwi}")

    # print("Test AUC - diffusion is:{:.3f}".format(AUC_test_diff))
    
    # fpr_diff, tpr_diff, _ = metrics.roc_curve(labels_diff, raw_preds_test_diff)
    fpr_dwi, tpr_dwi, _ = metrics.roc_curve(labels_dwi, raw_preds_test_dwi)

    plt.title('Receiver Operating Characteristic')
    # plt.plot(fpr_diff, tpr_diff, 'b', label = 'AUC diff = %0.2f' % AUC_test_diff, c= 'red')
    plt.plot(fpr_dwi, tpr_dwi, 'b', label = 'AUC DWI= %0.2f' % AUC_test_dwi, c= 'blue')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1.02])
    plt.ylim([0, 1.02])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    save_png_file = os.path.join(config_dwi['model_args']['rundir'], "Test_ROC_Curve_fastMRI_prostate.png")
    print("figure saved at", save_png_file)
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
    parser.add_argument('--config_file_dwi', type=str, required=True)           # config file which has all the training inputs
    # parser.add_argument('--config_file_diff', type=str, required=True)         # config file which has all the training inputs
    parser.add_argument('--index_seed', type=int)                               # Optional: Seed number for reproducibility for all numpy, random, torch
    parser.add_argument('--concat_t2w', type=str2bool, required=True, help='Set to True or False to specify whether to concatenate ADC as an additional channel.')
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
        with open(args_con.config_file_dwi) as f:
            args_dwi = yaml.load(f, Loader=yaml.UnsafeLoader)  

        # with open(args_con.config_file_diff) as f:
        #     args_diff = yaml.load(f, Loader=yaml.UnsafeLoader)  

        args_dwi['seed'] = seed_select
        args_dwi['concat_mask'] = args_con.concat_mask
        args_dwi['concat_t2w'] = args_con.concat_t2w
        args_dwi['focal_loss'] = args_con.focal_loss
        args_dwi['use_2_5d'] = args_con.use_2_5d
        print(args_dwi)

        main_fol_dwi = args_dwi["results_fol"]
        # main_fol_dwi = args_diff["results_fol"]
        subfolder = 'adc'  # Always include 't2w' as it's the base modality

        if args_dwi['concat_t2w']:
            subfolder += '_t2w'

        if args_dwi['concat_mask']:
            subfolder += '_mask'

        if args_dwi['use_2_5d']:
            subfolder += '_use_2_5d'

        args_dwi['model_args']['rundir'] = os.path.join(main_fol_dwi, subfolder, args_dwi['model_args']['rundir'] + '_SEED_' + str(seed_select))
        print("Model rundir DWI:{}".format(args_dwi['model_args']['rundir']))
        # args_diff['model_args']['rundir'] = os.path.join(main_fol_dwi, args_diff['model_args']['rundir'] + '_SEED_' + str(seed_select)) 

        # print("Model rundir diffusion:{}".format(args_diff['model_args']['rundir']))   

        torch.manual_seed(seed_select)                                           
        torch.cuda.manual_seed(seed_select)                               
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed_select)

        run_inference(args_dwi)

# %%