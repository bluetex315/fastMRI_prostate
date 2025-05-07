import argparse
import os
import yaml
import torch
import numpy as np
import pandas as pd
from sklearn import metrics

# import your data loader and model
from utils.custom_data_t2w_mask_adc import load_data
from model.model import ConvNext_model


def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes','true','t','y','1'): return True
    if v.lower() in ('no','false','f','n','0'): return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def evaluate(model, loader, device):
    """
    Run inference on `loader` and compute metrics.
    Returns dict with keys: auc, accuracy, recall, f1, conf_matrix.
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

    outs   = torch.cat(all_out, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy().astype(int)
    probs  = torch.softmax(outs, dim=1).numpy()
    preds  = np.argmax(probs, axis=1)

    res = {}
    try:
        res['auc'] = metrics.roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except ValueError:
        res['auc'] = float('nan')
    res['accuracy']  = metrics.accuracy_score(labels, preds)
    res['recall']    = metrics.recall_score(labels, preds, average='macro')
    res['precision'] = metrics.precision_score(labels, preds, average='macro')
    res['f1']        = metrics.f1_score(labels, preds, average='macro')
    res['conf_matrix'] = metrics.confusion_matrix(labels, preds)
    return res


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation of best_model_auc across multiple seeds"
    )
    parser.add_argument('--config_file', type=str, required=True,
                        help='YAML file with data and model settings (including results_fol)')
    parser.add_argument('--index_seed', type=int,
                        help='If set, only evaluate this seed index (0-based)')
    parser.add_argument('--concat_mask', type=str2bool, required=True,
                        help='Whether to concatenate gland mask as extra channel')
    parser.add_argument('--concat_adc', type=str2bool, required=True,
                        help='Whether to concatenate ADC as extra channel')
    parser.add_argument('--focal_loss', type=str2bool, default=False,
                        help='Whether to use focal loss instead of BCE')
    parser.add_argument('--ddp', action='store_true',
                        help='Whether training used DDP (affects loading)')
    parser.add_argument('--saveims', action='store_true',
                        help='If set, will dump sample images from test loader')
    parser.add_argument('--saveims_format', nargs='+', default=['png'],
                        help='Formats to save images: png, nifti, npz')
    args = parser.parse_args()

    # load config
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # apply flags
    config['ddp'] = args.ddp
    config['concat_mask'] = args.concat_mask
    config['concat_adc'] = args.concat_adc
    config['focal_loss'] = args.focal_loss
    config['training']['saveims'] = args.saveims
    config['training']['saveims_format'] = args.saveims_format

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # collect seed folders
    results_root = config['results_fol']
    seeds = sorted(d for d in os.listdir(results_root) if d.startswith('SEED'))
    seeds_nums = [int(s.split('_',1)[1]) for s in seeds]

    if not seeds:
        raise RuntimeError(f"No SEED_xxx folders in {results_root}")
    if args.index_seed is not None:
        seeds = [seeds[args.index_seed]]

    all_aucs = []
    all_accs = []
    all_recalls = []
    all_precisions = []
    all_f1s = []

    summary = []

    for seed in seeds_nums:

        seed_dir = os.path.join(results_root, "SEED_"+str(seed))
        print(seed_dir)
        print(f"\n=== Seed {seed} ===")

        # set rundir for saveims
        config['seed'] = seed
        config['model_args']['rundir'] = seed_dir

        # load test data
        _, _, test_loader = load_data(
            config=config,
            datapath=config['data']['fake_datapath'],
            labelpath=config['data']['labelpath'],
            gland_maskpath=config['data']['glandmask_path'],
            norm_type=int(config['data']['norm_type']),
            augment=False,
            saveims=config['training']['saveims'],
            saveims_format=config['training']['saveims_format'],
            rundir=seed_dir,
            rank=0,
            world_size=1
        )

        # build and load model
        model = ConvNext_model(config)
        ckpt = os.path.join(seed_dir, 'checkpoints', 'best_model_auc.pth')
        if not os.path.exists(ckpt):
            print(f"  MISSING: {ckpt}")
            continue
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        model.to(device)

        # evaluate
        res = evaluate(model, test_loader, device)
        print()
        print("*"*25+"Evaluating"+"*"*25)
        print(f"  AUC      : {res['auc']:.4f}")
        print(f"  Accuracy : {res['accuracy']:.4f}")
        print(f"  Recall   : {res['recall']:.4f}")
        print(f"  Precision: {res['precision']:.4f}")
        print(f"  F1 Score : {res['f1']:.4f}")
        print(f"  Confusion Matrix:\n{res['conf_matrix']}")
        print("*"*25+"finish"+"*"*25)
        print()

        all_aucs.append(res['auc'])
        all_accs.append(res['accuracy'])
        all_recalls.append(res['recall'])
        all_precisions.append(res['precision'])
        all_f1s.append(res['f1'])

        summary.append({
            'Seed': seed,
            'AUC': res['auc'],
            'ACC': res['accuracy'],
            'recall': res['recall'],
            'precision': res['precision'],
            'f1': res['f1']
        })
    
    if all_aucs:
        print(f"\nOverall AUC: {np.mean(all_aucs):.4f} ± {np.std(all_aucs):.4f}")
    if all_accs:
        print(f"Overall Accuracy: {np.mean(all_accs):.4f} ± {np.std(all_accs):.4f}")
    if all_recalls:
        print(f"Overall Recall: {np.mean(all_recalls):.4f} ± {np.std(all_recalls):.4f}")
    if all_precisions:
        print(f"Overall Precision: {np.mean(all_precisions):.4f} ± {np.std(all_precisions):.4f}")
    if all_f1s:
        print(f"Overall F1: {np.mean(all_f1s):.4f} ± {np.std(all_f1s):.4f}")

    # write CSV for real metrics only using pandas
    df = pd.DataFrame(summary)
    df = df.round({'AUC': 4, 'ACC': 4, 'recall': 4, 'precision': 4, 'f1': 4})
    
    mean_auc = df['AUC'].mean()
    std_auc = df['AUC'].std()
    mean_acc = df['ACC'].mean()
    std_acc = df['ACC'].std()
    mean_recall = df['recall'].mean()
    std_recall = df['recall'].std()
    mean_precision = df['precision'].mean()
    std_precision = df['precision'].std()
    mean_f1 = df['f1'].mean()
    std_f1 = df['f1'].std()
    
    summary_row = {
        'Seed': 'Mean±Std',
        'AUC': f"{mean_auc:.4f}±{std_auc:.4f}",
        'ACC': f"{mean_acc:.4f}±{std_acc:.4f}",
        'recall': f"{mean_recall:.4f}±{std_recall:.4f}",
        'precision': f"{mean_precision:.4f}±{std_precision:.4f}",
        'f1': f"{mean_f1:.4f}±{std_f1:.4f}"
    }
    
    summary_df = pd.DataFrame([summary_row])
    df = pd.concat([df, summary_df], ignore_index=True)

    date_tag = os.path.basename(results_root.rstrip(os.sep))
    csv_filename = f"{date_tag}_evaluation_summary.csv"
    csv_path = os.path.join(results_root, csv_filename)
    df.to_csv(csv_path, index=False)
    print(f"Wrote summary CSV → {csv_path}")


if __name__ == '__main__':
    main()
