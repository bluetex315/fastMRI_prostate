import argparse
import numpy as np
import os
import torch
from sklearn import metrics
# from utils.load_fastmri_data_convnext_t2 import load_data
from utils.custom_data_t2w_mask_adc import load_data
from model.model import ConvNext_model
from utils.pytorchtools import EarlyStopping
from model.extra_model_utils import get_optim_sched, get_lr
from shutil import copyfile
from torch.utils.tensorboard import SummaryWriter
import yaml
import pickle
from tqdm import tqdm

def train(model, optimizer, scheduler, train_loader, device):
    """
    Train the ConvNext model for one epoch.

    Parameters:
    - model: The ConvNext model.
    - optimizer: The PyTorch optimizer.
    - scheduler: The PyTorch learning rate scheduler.
    - train_loader: The PyTorch DataLoader for the training set.
    - device: The device (CPU or GPU) on which to perform the training.
    - drop_factor: The dropout factor.

    Returns:
    - auc (float): The area under the ROC curve.
    - current_lr (float): The current learning rate.
    - current_loss (float): The current loss.
    - labels (Tensor): Concatenated ground truth labels.
    - raw_preds (Tensor): Concatenated raw predictions.
    """

    total_loss_train, total_num, all_out, all_labels = 0.0, 0,  [], []  

    train_loader_tqdm = tqdm(train_loader, desc="Training", unit="batch")

    for _, (image, target) in enumerate(train_loader_tqdm):
        image = image.to(device)
        target = torch.flatten(target.to(device)) 

        optimizer.zero_grad()                                            
        out = model(image)                                                
        out = torch.flatten(out)     

        loss = train_loader.dataset.weighted_loss(out, target)          
        loss.backward()                                                
        optimizer.step()                                               

        total_loss_train += loss.item()                                 
        all_out.append(out)                                             
        all_labels.append(target)                                       
        total_num += 1                                                  

    all_labels_npy = torch.cat(all_labels).detach().cpu().numpy().astype(np.int32)   
    all_preds_npy = torch.sigmoid(torch.cat(all_out)).detach().cpu().numpy()         
    # Convert raw predictions to binary predictions
    binary_preds_npy = (all_preds_npy > 0.5).astype(int)

    auc = metrics.roc_auc_score(all_labels_npy, all_preds_npy)                      
    
    accuracy = metrics.accuracy_score(all_labels_npy, binary_preds_npy)
    recall = metrics.recall_score(all_labels_npy, binary_preds_npy)
    f1 = metrics.f1_score(all_labels_npy, binary_preds_npy)

    conf_matrix = metrics.confusion_matrix(all_labels_npy, binary_preds_npy)

    current_loss = total_loss_train/total_num                           
    current_lr = get_lr(optimizer)                                        
    scheduler.step()                                                      

    return auc, current_lr, current_loss, accuracy, recall, f1, conf_matrix, torch.cat(all_labels), torch.cat(all_out)

def val(model, val_loader, device):
    """
    Validate the ConvNext model on the validation set.

    Parameters:
    - model: The ConvNext model.
    - val_loader: The PyTorch DataLoader for the validation set.
    - device: The device (CPU or GPU) on which to perform the validation.
    - drop_factor: The dropout factor.

    Returns:
    - auc_val (float): The area under the ROC curve on the validation set.
    - current_loss (float): The current loss on the validation set.
    - labels_validation (Tensor): Concatenated ground truth labels from the validation set.
    - raw_preds_validation (Tensor): Concatenated raw predictions from the validation set.
    """
    total_loss_val, total_num_val, all_out, all_labels_val = 0.0, 0,  [], []  
    model.eval()   

    val_loader_tqdm = tqdm(val_loader, desc="Validating", unit="batch")  
        
    with torch.no_grad():                                                    
        for _, (image, target) in enumerate(val_loader_tqdm):
            image = image.to(device)
            # print(image.shape)
            target = torch.flatten(target.to(device)) 
            out = model(image)                                                
            out = torch.flatten(out)                                          
            loss = val_loader.dataset.weighted_loss(out, target)              
            
            all_out.append(out)                       
            all_labels_val.append(target)              
            total_loss_val += loss.item()              
            total_num_val += 1                         

    all_labels_npy = torch.cat(all_labels_val).detach().cpu().numpy().astype(np.int32) 
    all_preds_npy = torch.sigmoid(torch.cat(all_out)).detach().cpu().numpy()           
    binary_preds_npy = (all_preds_npy > 0.5).astype(int)

    auc_val = metrics.roc_auc_score(all_labels_npy, all_preds_npy)  

    accuracy = metrics.accuracy_score(all_labels_npy, binary_preds_npy)
    recall = metrics.recall_score(all_labels_npy, binary_preds_npy)
    f1 = metrics.f1_score(all_labels_npy, binary_preds_npy)
    
    conf_matrix = metrics.confusion_matrix(all_labels_npy, binary_preds_npy)

    current_loss = total_loss_val/total_num_val                                         
        
    return auc_val, current_loss, accuracy, recall, f1, conf_matrix, torch.cat(all_labels_val), torch.cat(all_out)


def train_network(config):
    """
    Train the ConvNext model based on the provided configuration.

    Parameters:
    - config (dict): Configuration parameters for training the ConvNext model.
    """
    early_stopping = EarlyStopping(patience=config['model_args']['patience'], verbose=True)      
    use_cuda = torch.cuda.is_available()                               
    device = torch.device("cuda" if use_cuda else "cpu")                
    print('Found this device:{}'.format(device))
    
    train_loader, valid_loader, test_loader = load_data(
        config,
        config['data']['datapath'], 
        config["data"]["labelpath"],
        config["data"]["glandmask_path"], 
        int(config['data']['norm_type']),  
        config['training']['augment'], 
        config['training']['saveims'], 
        config['model_args']['rundir']
    )

    print('Lengths: Train:{}, Val:{}, Test:{}'.format(len(train_loader), len(valid_loader), len(test_loader)))  
    
    model = ConvNext_model(config)
    model.to(device)
    print('Number of model parameters:{}'.format(sum(p.numel() for p in model.parameters())))
    optimizer, scheduler, scheduler2 = get_optim_sched(model, config) 

    dirin = config['model_args']['rundir']
    writer = SummaryWriter(log_dir = config['model_args']['rundir'])  
    
    saver = dict()                                      
    for e in range(config['training']['max_epochs']):  
        model.train()                                 
        AUC_train, current_LR, current_loss_train, acc_train, recall_train, f1_train, conf_matrix_train, labels_train, raw_preds_train = train(model, optimizer, scheduler, train_loader, device)       
        AUC_val, current_loss_val, acc_val, recall_val, f1_val, conf_matrix_val, labels_validation, raw_preds_validation  = val(model, valid_loader, device)     

        early_stopping(current_loss_val, model)       
        
        if early_stopping.early_stop:                 
            print("--Early stopping!!!--")
            break                                    
        if config['model_args']['scheduler_plat_loss']:
            scheduler2.step(current_loss_val)
        elif config['model_args']['scheduler_plat_auc']:
            scheduler2.step(AUC_val)
        else:
            pass    
        scheduler.step()

        writer.add_scalar("Loss/Train", current_loss_train, e)    
        writer.add_scalar("Loss/Validation", current_loss_val, e)  
        writer.add_scalar("AUC/Training", AUC_train, e)            
        writer.add_scalar("AUC/Validation", AUC_val, e)            
        writer.add_scalar("Learning Rate", current_LR, e)          
        
        print('Current epoch: {}, '
            'Train_loss: {:.3f}, '
            'Val_loss: {:.3f}, \n'
            'Train_ACC: {:.3f}, '
            'Val_ACC: {:.3f}, '
            'Train_AUC: {:.3f}, '
            'Val_AUC: {:.3f}, '
            'Train_Recall: {:.3f}, '
            'Val_Recall: {:.3f}, '
            'LR: {}'.format(e, current_loss_train, current_loss_val, acc_train, acc_val, AUC_train, AUC_val, recall_train, recall_val, current_LR))

        print(f"Confusion Matrix (Train):\n{conf_matrix_train}")
        print(f"Confusion Matrix (Val):\n{conf_matrix_val}")

        saver[e] = dict()
        saver[e]['val_preds'] = raw_preds_validation
        saver[e]['val_labels'] = labels_validation
        saver[e]['val_auc'] = AUC_val
        saver[e]['val_loss'] = current_loss_val
        saver[e]['train_preds'] = raw_preds_train
        saver[e]['train_labels'] = labels_train
        saver[e]['train_auc'] = AUC_train
        saver[e]['train_loss'] = current_loss_train

        if config['training']['save_model']:
            PATH = os.path.join(config['model_args']['rundir'],  'model_epoch_' + str(e) + '.pth') 
            torch.save(model.state_dict(), PATH)                                                  
    
    writer.close()                                                      
    savepath = os.path.join(dirin, 'model_outputs_raw.pkl')            
    with open(savepath, 'wb') as f:                                   
        pickle.dump(saver, f)

def get_parser():
    """
    Create an argument parser for the main script.

    Returns:
    - parser: The argparse.ArgumentParser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)           # config file which has all the training inputs
    parser.add_argument('--index_seed', type=int, required=True)            # Seed number for reproducibility for all numpy, random, torch
    parser.add_argument('--concat_mask', type=bool, required=True, help='Set to True or False to specify whether to concatenate gland mask as an additional channel.')
    parser.add_argument('--concat_adc', type=bool, required=True, help='Set to True or False to specify whether to concatenate ADC as an additional channel.')
    return parser


if __name__ == '__main__':
    """
    Main script for training the ConvNext model.
    """
    args_con = get_parser().parse_args() 
    seed_list = [10383, 44820, 238, 3939, 74783, 92938, 143, 2992, 7373, 988]           
    seed_select =  seed_list[args_con.index_seed]                                      
    
    with open(args_con.config_file) as f:
        args = yaml.load(f, Loader=yaml.UnsafeLoader) 

    args['concat_mask'] = args_con.concat_mask
    args['concat_adc'] = args_con.concat_adc

    main_fol = args["results_fol"]
    args['model_args']['rundir'] = os.path.join(main_fol, args['model_args']['rundir'] + '_SEED_' + str(seed_select)) 
    print("Model rundir:{}".format(args['model_args']['rundir']))
    if not os.path.isdir(args['model_args']["rundir"]):
        os.makedirs(os.path.join(args['model_args']["rundir"]))                                   

    copyfile(args_con.config_file, os.path.join(args['model_args']['rundir'], 'params.txt'))
    
    torch.manual_seed(seed_select)                                           
    torch.cuda.manual_seed(seed_select)                               
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_select)

    print(args)
    train_network(args)

# %%
