import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.distributed as dist
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
import matplotlib.pyplot as plt
from datetime import datetime


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

    total_loss_train, total_num, all_out, all_labels = 0.0, 0, [], []  

    for images, targets, patient_ids, slice_ids in tqdm(train_loader, desc="Training", disable=(dist.is_initialized() and dist.get_rank()!=0)):
        images = images.to(device)
        targets = torch.flatten(targets.to(device))

        optimizer.zero_grad()                                            
        out = model(images)                                         
        # out = torch.flatten(out)     

        loss = train_loader.dataset.loss(out, targets) 
          
        loss.backward()                                                
        optimizer.step()                                               

        total_loss_train += loss.item()                                 
        all_out.append(out.detach())                                             
        all_labels.append(targets.detach())                                       
        total_num += 1                                                  

    # STEP 1: concatenate local outputs & labels
    local_outs = torch.cat(all_out, dim=0)  # [N_local, C]
    local_labels = torch.cat(all_labels, dim=0)  # [N_local]
    
    # STEP 2: if DDP, gather from all ranks
    if dist.is_initialized() and dist.get_world_size() > 1:
        world_size = dist.get_world_size()

        # 2a) gather sizes so we know how many examples each rank had
        local_size = torch.tensor([local_labels.size(0)], device=device)

        sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(sizes, local_size)

        max_size = max([int(sz.item()) for sz in sizes])

        # 2b) pad to max_size
        padded_labels = torch.zeros(max_size, device=device, dtype=local_labels.dtype)
        padded_labels[:local_labels.size(0)] = local_labels

        padded_outs = torch.zeros(max_size, local_outs.size(1), device=device, dtype=local_outs.dtype)
        padded_outs[:local_outs.size(0)] = local_outs

        # 2c) gather padded tensors
        gathered_labels = [torch.zeros_like(padded_labels) for _ in range(world_size)]
        gathered_outs = [torch.zeros_like(padded_outs) for _ in range(world_size)]
        dist.all_gather(gathered_labels, padded_labels)
        dist.all_gather(gathered_outs, padded_outs)

        # 2d) un-pad and concat
        labels_list, outs_list = [], []
        for rank_idx, sz in enumerate(sizes):
            n = int(sz.item())
            labels_list.append(gathered_labels[rank_idx][:n].cpu())
            outs_list.append(gathered_outs[rank_idx][:n].cpu())
        all_labels_tensor = torch.cat(labels_list, dim=0)
        all_outs_tensor = torch.cat(outs_list, dim=0)

    else:
        # single-GPU or non-distributed
        all_labels_tensor = local_labels.cpu()
        all_outs_tensor = local_outs.cpu()

    # STEP 3: only rank 0 computes metrics & returns them
    if not dist.is_initialized() or dist.get_rank() == 0:
        # convert to numpy
        all_labels_npy = all_labels_tensor.detach().cpu().numpy().astype(np.int32)
        all_probs_npy = torch.softmax(all_outs_tensor, dim=1).detach().cpu().numpy()
        all_preds_npy = np.argmax(all_probs_npy, axis=1)

        print("line 111", all_probs_npy, all_probs_npy.shape)

        try:
            auc = metrics.roc_auc_score(all_labels_npy, all_probs_npy, multi_class='ovr', average='macro')  

        except ValueError as e:
            # e.g. "Number of classes in y_true not equal …"
            print(f"Skipping Val AUC: {e}")
            auc = 0         
        
        accuracy = metrics.accuracy_score(all_labels_npy, all_preds_npy)
        recall = metrics.recall_score(all_labels_npy, all_preds_npy, average='macro')
        f1 = metrics.f1_score(all_labels_npy, all_preds_npy, average='macro')

        conf_matrix = metrics.confusion_matrix(all_labels_npy, all_preds_npy)

        current_loss = total_loss_train/total_num                           
        current_lr = get_lr(optimizer)                                        
        scheduler.step()                                                      

        return (auc, current_lr, current_loss, accuracy, recall, f1, conf_matrix, torch.cat(all_labels), torch.cat(all_out))
    
    else:
        return (None,)*9

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

    model.eval()

    total_loss_val, total_num_val, all_out, all_labels_val = 0.0, 0, [], []  
 
    val_loader_tqdm = tqdm(val_loader, desc="Validating", unit="batch")  
        
    with torch.no_grad():                                                    
        for images, targets, patient_ids, slice_ids in tqdm(val_loader, desc="Validating", disable=(dist.is_initialized() and dist.get_rank() != 0)):
            images = images.to(device)
            targets = torch.flatten(targets.to(device)) 

            out = model(images)                                                
            criterion = nn.CrossEntropyLoss()
            loss = criterion(out, targets)                                            

            total_loss_val += loss.item()              
            total_num_val += 1        

            all_out.append(out.detach())                       
            all_labels_val.append(targets.detach())              

    # 1) concatenate local tensors
    local_outs = torch.cat(all_out, dim=0)  # [N_local, C]
    local_labels = torch.cat(all_labels_val, dim=0)  # [N_local]

    # 2) if in DDP, gather from all ranks
    if dist.is_initialized() and dist.get_world_size() > 1:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # a) gather sizes
        local_size = torch.tensor([local_labels.size(0)], device=device)
        sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(sizes, local_size)
        max_size = max(int(sz.item()) for sz in sizes)

        # b) pad to max_size
        padded_labels = torch.zeros(max_size, device=device, dtype=local_labels.dtype)
        padded_labels[:local_labels.size(0)] = local_labels
        padded_outs = torch.zeros(max_size, local_outs.size(1), device=device, dtype=local_outs.dtype)
        padded_outs[:local_outs.size(0)] = local_outs

        # c) all_gather padded tensors
        gathered_labels = [torch.zeros_like(padded_labels) for _ in range(world_size)]
        gathered_outs = [torch.zeros_like(padded_outs) for _ in range(world_size)]
        dist.all_gather(gathered_labels, padded_labels)
        dist.all_gather(gathered_outs, padded_outs)

        # d) un‐pad and concat
        labels_list, outs_list = [], []
        for i, sz in enumerate(sizes):
            n = int(sz.item())
            labels_list.append(gathered_labels[i][:n].cpu())
            outs_list.append(gathered_outs[i][:n].cpu())

        all_labels_tensor = torch.cat(labels_list, dim=0)
        all_outs_tensor = torch.cat(outs_list, dim=0)
    else:
        # single‐GPU or non‐distributed
        all_labels_tensor = local_labels.cpu()
        all_outs_tensor = local_outs.cpu()

    # 3) only rank 0 (or single‐GPU) does numpy conversion & sklearn metrics
    if not dist.is_initialized() or dist.get_rank() == 0:
        # preserve your original lines
        all_labels_npy = all_labels_tensor.detach().cpu().numpy().astype(np.int32)
        all_probs_npy = torch.softmax(all_outs_tensor, dim=1).detach().cpu().numpy()
        all_preds_npy = np.argmax(all_probs_npy, axis=1)

        print("line 226", all_preds_npy, all_preds_npy.shape)

        try:
            auc_val = metrics.roc_auc_score(all_labels_npy, all_probs_npy, multi_class='ovr', average='macro')  

        except ValueError as e:
            # e.g. "Number of classes in y_true not equal …"
            print(f"Skipping Val AUC: {e}")
            auc_val = 0

        accuracy = metrics.accuracy_score(all_labels_npy, all_preds_npy)
        recall = metrics.recall_score(all_labels_npy, all_preds_npy, average='macro')
        f1 = metrics.f1_score(all_labels_npy, all_preds_npy, average='macro')
        
        conf_matrix = metrics.confusion_matrix(all_labels_npy, all_preds_npy)

        current_loss = total_loss_val/total_num_val                                         
            
        return (auc_val, current_loss, accuracy, recall, f1, conf_matrix, torch.cat(all_labels_val), torch.cat(all_out))
    
    else:
        return (None,)*8


def train_network(config, rank, world_size, is_main):
    """
    Train the ConvNext model based on the provided configuration.

    Parameters:
    - config (dict): Configuration parameters for training the ConvNext model.
    """

    # device setup
    if config['ddp']:
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        print("train_network line260 device", device)
        print()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if is_main:
        print(f"[Rank {rank}] Using device {device}")
    
    train_loader, valid_loader, test_loader = load_data(
        config=config,
        datapath=config['data']['fake_datapath'], 
        labelpath=config["data"]["labelpath"],
        gland_maskpath=config["data"]["glandmask_path"], 
        norm_type=int(config['data']['norm_type']),  
        augment=config['training']['augment'], 
        saveims=config['training']['saveims'], 
        saveims_format=config['training']['saveims_format'],
        rundir=config['model_args']['rundir'],
        rank=rank,
        world_size=world_size
    )

    print('Lengths of DataLoader: Train:{}, Val:{}, Test:{}'.format(len(train_loader), len(valid_loader), len(test_loader)))  
    
    model = ConvNext_model(config).to(device)

    if config['ddp']:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    if is_main:
        print("Number of params:", sum(p.numel() for p in model.parameters()))

    if config['model_args']['warm_up']:
        optimizer, scheduler, warmup_scheduler, scheduler2 = get_optim_sched(model, config) 
    else:
        optimizer, scheduler, scheduler2 = get_optim_sched(model, config) 

    early_stopping = EarlyStopping(patience=config['training']['patience'], verbose=is_main)  
    
    dirin = config['model_args']['rundir']
    loss_dir = os.path.join(dirin, "checkpoints", "best_loss_models")
    auc_dir = os.path.join(dirin, "checkpoints", "best_auc_models")
    if is_main:
        os.makedirs(loss_dir, exist_ok=True)
        os.makedirs(auc_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=config['model_args']['rundir'])

    saver = dict()    
    
    best_val_loss = float('inf')      # for saving 'best.pth' according to loss
    best_loss_epoch = -1        
    
    best_val_auc = -float("inf")        # for saving 'best.pth' according to auc
    best_auc_epoch = -1

    best_loss_models = []               # for saving top 3 best pth according to loss
    prev_loss_epochs = set()
    saved_loss_info = {}

    best_auc_models = []                # for saving top 3 best pth according to auc
    prev_auc_epochs = set()
    saved_auc_info = {}

    for e in range(config['training']['max_epochs']):  
        if config['ddp']:
            train_loader.sampler.set_epoch(e)
            valid_loader.sampler.set_epoch(e)
            
        model.train()                                 
        AUC_train, current_LR, current_loss_train, acc_train, recall_train, f1_train, conf_matrix_train, labels_train, raw_preds_train = train(model, optimizer, scheduler, train_loader, device)       
        AUC_val, current_loss_val, acc_val, recall_val, f1_val, conf_matrix_val, labels_validation, raw_preds_validation = val(model, valid_loader, device)     

        if is_main and current_loss_val is not None:
            if config['training']['early_stop']:
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
                # raise ValueError('An unsupported type of scheduler.')    
        
        if config['model_args']['warm_up']:
            if e < config['model_args']['warmup_epochs']:
                warmup_scheduler.step()
            else:
                scheduler.step()
        else:
            scheduler.step()

        if is_main:
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

            if config['training']['save_model_loss']:
                # 1) update the top‑3 list
                best_loss_models.append((current_loss_val, e))
                best_loss_models = sorted(best_loss_models, key=lambda x: x[0])[:3]
                new_best_epochs = set(epoch for _, epoch in best_loss_models)

                # 2) delete any old files that dropped out of top‑3
                to_delete = prev_loss_epochs - new_best_epochs
                for epoch_idx in to_delete:
                    old_loss_val = saved_loss_info.pop(epoch_idx, None)
                    old_fname = f"model_val_loss_{old_loss_val:.4f}_epoch_{epoch_idx}.pth"
                    old_path = os.path.join(dirin, 'checkpoints', 'best_loss_models', old_fname)
                    if os.path.exists(old_path):
                        os.remove(old_path)
                        print(f"[Epoch {e}] Removed old loss checkpoint → {old_path}")

                # 3) save any *new* epochs that entered top‑3
                to_save = new_best_epochs - prev_loss_epochs
                for epoch_idx in to_save:
                    loss_val = next(l for l,e in best_loss_models if e==epoch_idx)
                    fname = f"model_val_loss_{loss_val:.4f}_epoch_{epoch_idx}.pth"
                    ckpt_path = os.path.join(loss_dir, fname)
                    torch.save(model.state_dict(), ckpt_path)
                    saved_loss_info[epoch_idx] = loss_val
                    print(f"[Epoch {epoch_idx}] Saved loss checkpoint → {ckpt_path}")

                # 4) update your tracker
                prev_loss_epochs = new_best_epochs.copy()

                if current_loss_val < best_val_loss:
                    best_val_loss = current_loss_val
                    best_loss_epoch = e
                    path = os.path.join(dirin, "checkpoints", "best_model_loss.pth")
                    torch.save(model.state_dict(), path)
                    print(f"[Epoch {e}] Saved overall best loss model (Loss={current_loss_val:.4f}) → {path}")
            if config['training']['save_model_auc']:
                # 1) update the top‑3 list (largest AUC first)
                best_auc_models.append((AUC_val, e))
                best_auc_models = sorted(best_auc_models, key=lambda x: x[0], reverse=True)[:3]
                new_auc_epochs = set(epoch for _, epoch in best_auc_models)

                # 2) delete any old files that dropped out of top‑3
                to_delete = prev_auc_epochs - new_auc_epochs
                for epoch_idx in to_delete:
                    old_auc_val = saved_auc_info.pop(epoch_idx)
                    old_fname = f"model_val_auc_{old_auc_val:.4f}_epoch_{epoch_idx}.pth"
                    old_path = os.path.join(dirin, 'checkpoints', 'best_auc_models', old_fname)
                    if os.path.exists(old_path):
                        os.remove(old_path)
                        print(f"[Epoch {e}] Removed old AUC checkpoint → {old_path}")

                # 3) save new epochs that entered top‑3
                to_save = new_auc_epochs - prev_auc_epochs
                for epoch_idx in to_save:
                    auc_val = next(a for a, ep in best_auc_models if ep == epoch_idx)
                    fname = f"model_val_auc_{auc_val:.4f}_epoch_{epoch_idx}.pth"
                    ckpt_path = os.path.join(dirin, 'checkpoints', 'best_auc_models', fname)
                    torch.save(model.state_dict(), ckpt_path)
                    saved_auc_info[epoch_idx] = auc_val
                    print(f"[Epoch {epoch_idx}] Saved AUC checkpoint → {ckpt_path}")
                    
                # 4) update tracker
                prev_auc_epochs = new_auc_epochs.copy()

                # ── overall single best by AUC ──
                if AUC_val > best_val_auc:
                    best_val_auc   = AUC_val
                    best_auc_epoch = e
                    best_path = os.path.join(dirin, "checkpoints", "best_model_auc.pth")
                    torch.save(model.state_dict(), best_path)
                    print(f"[Epoch {e}] Saved overall best AUC model (AUC={AUC_val:.4f}) → {best_path}")
                    
            last_epoch = config['training']['max_epochs'] - 1
            if (config['training']['save_model_auc'] or config['training']['save_model_loss']) and e == last_epoch:
                final_ckpt = os.path.join(dirin, "checkpoints", f"model_final_epoch_{e}.pth")
                torch.save(model.state_dict(), final_ckpt)
                print(f"[Epoch {e}] Saved final epoch checkpoint → {final_ckpt}")

            # if current_loss_val < lowest_val_loss:
            #     lowest_val_loss = current_loss_val
            #     lowest_val_epoch = e

            #     if config['training']['save_ROC_AUC']:

            #         plt.figure()
            #         plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            #         plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            #         plt.xlabel('False Positive Rate')
            #         plt.ylabel('True Positive Rate')
            #         plt.title('Receiver Operating Characteristic (ROC)')
            #         plt.legend(loc='lower right')
            #         save_path = os.path.join(config['model_args']['rundir'], "roc_auc_{}_epoch_{}.png".format(roc_auc, e))
            #         plt.savefig(save_path)
            #         plt.close()

            #     if config['training']['save_model']:
            #         PATH = os.path.join(config['model_args']['rundir'],  'model_epoch_' + str(e) + '.pth') 
            #         torch.save(model.state_dict(), PATH)                                                  
    if is_main:
        writer.close()                                                      
        savepath = os.path.join(dirin, 'model_outputs_raw.pkl')            
        with open(savepath, 'wb') as f:                                   
            pickle.dump(saver, f)

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
    parser.add_argument('--config_file', type=str, required=True)           # config file which has all the training inputs
    parser.add_argument('--index_seed', type=int)                   # Seed number for reproducibility for all numpy, random, torch, if not provided, loop through all seeds
    parser.add_argument('--concat_mask', type=str2bool, required=True, help='Set to True or False to specify whether to concatenate gland mask as an additional channel.')
    parser.add_argument('--concat_adc', type=str2bool, required=True, help='Set to True or False to specify whether to concatenate ADC as an additional channel.')
    parser.add_argument('--focal_loss', type=str2bool, default=False, help='whether to use focal loss instead of weighted bce')
    parser.add_argument('--ddp', action='store_true', help='whether use ddp for training')
    return parser


def main_worker(rank, world_size, args):

    torch.cuda.set_device(rank)
    print(f"[Rank {rank}/{world_size}] 🚀 binding to GPU {rank} -> "
          f"current_device={torch.cuda.current_device()}")
    print()
    
    if args.ddp:
        # 1) DDP setup
        torch.cuda.set_device(rank)
        
        dist.init_process_group(backend='nccl',
                                init_method='tcp://127.0.0.1:29500',
                                rank=rank,
                                world_size=world_size)
        
        device = torch.device(f'cuda:{rank}')
        is_main = (rank == 0)

    else:
        # single-GPU / CPU path: never touch dist
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_main = True

    # 2) load & broadcast config
    with open(args.config_file) as f:
        base_cfg = yaml.load(f, Loader=yaml.UnsafeLoader)

    # 3) seed
    seed_list = [10383, 44820, 238, 3939, 74783, 92938, 143, 2992, 7373, 988]
    # Check if a specific seed index is provided
    if args.index_seed is not None:
        # Use the specific seed from the list
        seed_list = [seed_list[args.index_seed]]

    # Set the model directory based on the seed and time
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")   # e.g. "20250430_142532"
    if is_main:
        print(f"[MAIN]: Start Running on {ts}\n")

    # 4) keep track of other settings
    config = dict(base_cfg)
    config['concat_mask'] = args.concat_mask
    config['concat_adc'] = args.concat_adc
    config['focal_loss'] = args.focal_loss
    config['ddp'] = args.ddp

    # 5) Loop through all seeds
    for seed_select in seed_list:
        
        config['seed'] = seed_select

        main_fol = config["results_fol"]
        subfolder = 't2w'  # Always include 't2w' as it's the base modality

        if config['concat_adc']:
            subfolder += '_adc'
        if config['concat_mask']:
            subfolder += '_mask'
        
        config['model_args']['rundir'] = os.path.join(
            main_fol,
            subfolder,
            ts,
            f"SEED_{seed_select}"
        )
        print("Model rundir:{}".format(config['model_args']['rundir']))

        # # Create directory if it doesn't exist
        # if not os.path.isdir(config['model_args']["rundir"]):
        #     os.makedirs(os.path.join(config['model_args']["rundir"]))

        if is_main:
            rundir = config['model_args']['rundir']
            os.makedirs(rundir, exist_ok=True)
            # Copy config file to the new directory
            copyfile(args.config_file, os.path.join(rundir, 'params.txt'))

        # Set the random seed
        torch.manual_seed(seed_select + rank)
        torch.cuda.manual_seed(seed_select + rank)
        np.random.seed(seed_select + rank)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 5) finally call training loop
        print(config)
        train_network(config, rank, world_size, is_main)

    # 6) Cleanup
    if config['ddp']:
        dist.destroy_process_group()

if __name__ == "__main__":

    args = get_parser().parse_args()

    world_size = torch.cuda.device_count()

    print(f"\n[MAIN] <<<<<<<<<<<<<<<<<<<<<< world size: {world_size} >>>>>>>>>>>>>>>>>>>>>>>> \n")
    # if we have more than one, spawn that many processes
    if world_size > 1 and args.ddp:
        print("[MAIN] <<<<<<<<<<<<<<<<<<<<<< DDP TRAINING >>>>>>>>>>>>>>>>>>>>>>>>\n")
        torch.multiprocessing.spawn(
            main_worker,               # the function to run in each process
            args=(world_size, args),   # extra args passed to main_worker
            nprocs=world_size,         # number of processes = number of GPUs
            join=True
        )
    else:
        print("[MAIN] <<<<<<<<<<<<<<<<<<<<<< REGULAR TRAINING >>>>>>>>>>>>>>>>>>>>>>>>\n")
        # single‐GPU or CPU: just call main_worker once with rank=0
        args.ddp = False        # cannot use ddp on only 1 card, no matter the setting is
        main_worker(rank=0, world_size=1, args=args)
