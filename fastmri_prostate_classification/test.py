#!/usr/bin/env python3
"""
Test script for ConvNext prostate MRI classification.

Loads a trained checkpoint, runs inference on the test set, and prints evaluation metrics.
"""
import argparse
import os
import yaml
import torch
import numpy as np
from sklearn import metrics

# import your data loader and model
from utils.custom_data_t2w_mask_adc import load_data
from model.model import ConvNext_model


def evaluate(model, loader, device):
    """
    Run inference on `loader` and compute metrics.
    Returns a dict of {auc, accuracy, recall, f1, conf_matrix}.
    """
    model.eval()
    all_out, all_labels = [], []
    with torch.no_grad():
        for images, targets, _, _ in loader:
            images = images.to(device)
            targets = targets.flatten().to(device)
            out = model(images)
            all_out.append(out.cpu())
            all_labels.append(targets.cpu())

    outs = torch.cat(all_out, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy().astype(int)
    probs = torch.softmax(outs, dim=1).numpy()
    preds = np.argmax(probs, axis=1)

    results = {}
    try:
        results['auc'] = metrics.roc_auc_score(
            labels, probs, multi_class='ovr', average='macro'
        )
    except ValueError:
        results['auc'] = float('nan')
    results['accuracy'] = metrics.accuracy_score(labels, preds)
    results['recall'] = metrics.recall_score(labels, preds, average='macro')
    results['f1'] = metrics.f1_score(labels, preds, average='macro')
    results['conf_matrix'] = metrics.confusion_matrix(labels, preds)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained ConvNext model on the test set")
    parser.add_argument('--config_file', type=str, required=True, help='YAML file with data and model settings')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the .pth checkpoint file')
    parser.add_argument('--concat_mask', type=str2bool, required=True, help='Set to True or False to specify whether to concatenate gland mask as an additional channel.')
    parser.add_argument('--concat_adc', type=str2bool, required=True, help='Set to True or False to specify whether to concatenate ADC as an additional channel.')
    parser.add_argument('--index_seed', type=int, help='Seed number for reproducibility for all numpy, random, torch, if not provided, loop through all seeds')
    parser.add_argument('--saveims', action='store_true', help='If set, will dump sample images from test loader')
    parser.add_argument('--saveims_format', nargs='+', default=['png'], help='List of formats to save images: png, nifti, npz')
    args = parser.parse_args()

    # load config
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # disable DDP for testing
    config['ddp'] = False
    config['training']['saveims'] = args.saveims
    config['training']['saveims_format'] = args.saveims_format

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # load data
    train_loader, val_loader, test_loader = load_data(
        config=config,
        datapath=config['data']['fake_datapath'],
        labelpath=config['data']['labelpath'],
        gland_maskpath=config['data']['glandmask_path'],
        norm_type=int(config['data']['norm_type']),
        augment=False,
        saveims=config['training']['saveims'],
        saveims_format=config['training']['saveims_format'],
        rundir=config['model_args']['rundir'],
        rank=0,
        world_size=1
    )

    # build model and load weights
    model = ConvNext_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)

    # evaluate on test set
    print("Running evaluation on test set...")
    results = evaluate(model, test_loader, device)

    # print results
    print("Test Results:")
    print(f"AUC      : {results['auc']:.4f}")
    print(f"Accuracy : {results['accuracy']:.4f}")
    print(f"Recall   : {results['recall']:.4f}")
    print(f"F1 Score : {results['f1']:.4f}")
    print("Confusion Matrix:")
    print(results['conf_matrix'])


if __name__ == '__main__':
    main()
