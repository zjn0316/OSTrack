import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import _init_paths
from lib.train.data import opencv_loader
from lib.train.dataset import OTB100UWB
from lib.models.ugtrack import build_ugtrack


def parse_args():
    parser = argparse.ArgumentParser(description='UGTrack Stage-1 evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to checkpoint.pth')
    parser.add_argument('--config', type=str, required=True,
                        help='path to yaml config file')
    parser.add_argument('--split', type=str, default='test',
                        help='dataset split to evaluate on (default: test)')
    parser.add_argument('--seq_len', type=int, default=10,
                        help='UWB sequence length (must match training)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='save plots to this directory')
    return parser.parse_args()


# ============================================
# Metric helpers
# ============================================

def compute_uv_error(pred_uv, gt_uv):
    """L2 distance between pred_uv [N, 2] and gt_uv [N, 2] (normalized [0,1])."""
    return torch.norm(pred_uv - gt_uv, dim=-1).cpu().numpy()


def compute_uv_pred_auc(errors, threshold_max=0.50, step=0.01):
    """Success rate at each threshold, then AUC (trapezoidal rule)."""
    thresholds = np.arange(step, threshold_max + step, step)
    success_rates = np.array([(errors < t).mean() for t in thresholds])
    norm_auc = np.trapz(success_rates, thresholds) / threshold_max
    return thresholds, success_rates, norm_auc


def _roc_auc(labels, scores):
    """Manual ROC AUC via sorting (equivalent to sklearn's roc_auc_score)."""
    order = np.argsort(scores)
    labels_sorted = labels[order]
    pos_count = labels_sorted.sum()
    neg_count = len(labels_sorted) - pos_count
    if pos_count == 0 or neg_count == 0:
        return float('nan')
    rank_sum = (labels_sorted == 1).nonzero()[0].sum()
    return (rank_sum - pos_count * (pos_count - 1) / 2) / (pos_count * neg_count)


def compute_conf_auc(conf_pred, errors, error_thresh=0.05):
    """ROC AUC: does conf_pred predict low UV error?"""
    labels = (errors < error_thresh).astype(np.float32)
    conf_scores = conf_pred.flatten()
    if len(np.unique(labels)) < 2:
        return float('nan')
    return _roc_auc(labels, conf_scores)


def compute_occlusion_auc(conf_pred, visible_flags):
    """ROC AUC: does conf_pred predict visible vs occluded?"""
    labels = visible_flags.astype(np.float32)
    conf_scores = conf_pred.flatten()
    if len(np.unique(labels)) < 2:
        return float('nan')
    return _roc_auc(labels, conf_scores)


def compute_losses(pred_uv, gt_uv, conf_logit, gt_conf):
    """Recompute L1 + BCEWithLogitsLoss for reference."""
    from torch.nn.functional import l1_loss, binary_cross_entropy_with_logits
    pred_loss = l1_loss(pred_uv, gt_uv[..., :2]).item()
    conf_loss = binary_cross_entropy_with_logits(conf_logit, gt_conf).item()
    return pred_loss, conf_loss, pred_loss + conf_loss


# ============================================
# Evaluation
# ============================================

def evaluate(checkpoint_path, config_path, split, seq_len, save_dir=None):
    # -------------------------------------------------------
    # Config & model
    # -------------------------------------------------------
    import importlib
    config_module = importlib.import_module('lib.config.ugtrack.config')
    cfg = config_module.cfg
    config_module.update_config_from_file(config_path)
    cfg.TRAIN.STAGE = 1

    net = build_ugtrack(cfg, training=False)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    missing, unexpected = net.load_state_dict(ckpt['net'], strict=False)
    print('Checkpoint: {}'.format(checkpoint_path))
    print('  missing_keys: {}'.format(missing))
    print('  unexpected_keys: {}'.format(unexpected))
    net.cuda()
    net.eval()

    # -------------------------------------------------------
    # Dataset
    # -------------------------------------------------------
    from lib.train.admin import env_settings
    dataset = OTB100UWB(root=env_settings().otb100_uwb_dir, image_loader=opencv_loader,
                        split=split, uwb_seq_len=seq_len)
    n_seqs = dataset.get_num_sequences()
    print('Evaluating on {} split, {} sequences'.format(split, n_seqs))

    # -------------------------------------------------------
    # Inference
    # -------------------------------------------------------
    coord_scale = float(cfg.DATA.SEARCH.SIZE)  # normalize to [0,1] like training

    all_pred_uv = []
    all_gt_uv = []
    all_conf_pred = []
    all_conf_logit = []
    all_gt_conf = []
    all_visible = []

    with torch.no_grad():
        for seq_id in range(n_seqs):
            seq_info = dataset.get_sequence_info(seq_id)
            n_frames = seq_info['bbox'].shape[0]
            visible = seq_info['visible'].cpu().numpy()

            for f_id in range(n_frames):
                search_uwb_seq = seq_info['uwb_seq'][f_id].unsqueeze(0).cuda().float()
                gt_uv = seq_info['uwb_gt'][f_id, :2].unsqueeze(0).cuda()
                gt_conf = seq_info['uwb_conf'][f_id].view(1, 1).cuda().float()

                # Normalize pixel coordinates to [0,1] like UWBStage1Dataset did
                search_uwb_seq = (search_uwb_seq / coord_scale).clamp(0.0, 1.0)
                gt_uv = (gt_uv / coord_scale).clamp(0.0, 1.0)

                out = net(search_uwb_seq=search_uwb_seq, stage=1)
                pred_uv = out['pred_uv']
                conf_logit = out['uwb_conf_logit']
                conf_pred = out['uwb_conf_pred']

                all_pred_uv.append(pred_uv.cpu())
                all_gt_uv.append(gt_uv.cpu())
                all_conf_pred.append(conf_pred.cpu())
                all_conf_logit.append(conf_logit.cpu())
                all_gt_conf.append(gt_conf.cpu())
                all_visible.append(visible[f_id])

    # -------------------------------------------------------
    # Aggregate
    # -------------------------------------------------------
    pred_uv = torch.cat(all_pred_uv, dim=0)
    gt_uv = torch.cat(all_gt_uv, dim=0)
    conf_pred = torch.cat(all_conf_pred, dim=0)
    conf_logit = torch.cat(all_conf_logit, dim=0)
    gt_conf = torch.cat(all_gt_conf, dim=0)
    visible_arr = np.array(all_visible)

    # Losses
    pred_loss, conf_loss, total_loss = compute_losses(pred_uv, gt_uv, conf_logit, gt_conf)
    print('\n========== Losses ==========')
    print('  Loss/uwb_total:  {:.5f}'.format(total_loss))
    print('  Loss/uwb_pred:   {:.5f}'.format(pred_loss))
    print('  Loss/uwb_conf:   {:.5f}'.format(conf_loss))

    # UV prediction AUC
    errors = compute_uv_error(pred_uv, gt_uv)
    thresholds, success_rates, uv_auc = compute_uv_pred_auc(errors)
    print('\n========== UV Prediction AUC ==========')
    print('  uv_pred_auc:      {:.4f}'.format(uv_auc))
    print('  mean L2 error:    {:.4f}'.format(errors.mean()))

    # Confidence AUC (predicting low UV error)
    conf_auc = compute_conf_auc(conf_pred, errors, error_thresh=0.05)
    print('\n========== Confidence AUC ==========')
    print('  conf_auc (err<0.05):  {:.4f}'.format(conf_auc))

    # Occlusion AUC (predicting visibility)
    occ_auc = compute_occlusion_auc(conf_pred, visible_arr)
    print('\n========== Occlusion AUC ==========')
    print('  occlusion_auc:    {:.4f}'.format(occ_auc))

    # -------------------------------------------------------
    # Plot
    # -------------------------------------------------------
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(checkpoint_path))[0]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].plot(thresholds, success_rates, linewidth=2)
        axes[0].fill_between(thresholds, success_rates, alpha=0.2)
        axes[0].axvline(0.05, color='gray', linestyle='--', alpha=0.5, label='thresh=0.05')
        axes[0].set_xlabel('Error threshold')
        axes[0].set_ylabel('Success rate')
        axes[0].set_title('UV Prediction AUC = {:.4f}'.format(uv_auc))
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[1].axvline(errors.mean(), color='red', linestyle='--',
                        label='mean={:.4f}'.format(errors.mean()))
        axes[1].set_xlabel('L2 error')
        axes[1].set_ylabel('Count')
        axes[1].set_title('UV Error Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        visible_errs = errors[visible_arr == 1]
        occluded_errs = errors[visible_arr == 0]
        if len(occluded_errs) > 0:
            axes[2].hist(visible_errs, bins=40, alpha=0.6, label='visible', color='green')
            axes[2].hist(occluded_errs, bins=40, alpha=0.6, label='occluded', color='red')
            axes[2].set_xlabel('L2 error')
            axes[2].set_ylabel('Count')
            axes[2].set_title('Error by Visibility')
            axes[2].legend()
        else:
            axes[2].hist(errors, bins=40, alpha=0.7, edgecolor='black')
            axes[2].set_xlabel('L2 error')
            axes[2].set_title('UV Error (no occlusion in split)')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(save_dir, '{}_uwb_eval.png'.format(basename))
        plt.savefig(plot_path, dpi=150)
        print('\nPlot saved to: {}'.format(plot_path))

    # -------------------------------------------------------
    # Summary line
    # -------------------------------------------------------
    print('\n========== SUMMARY ==========')
    print('{:.5f}  {:.5f}  {:.5f}  {:.4f}  {:.4f}  {:.4f}'.format(
        total_loss, pred_loss, conf_loss, uv_auc, conf_auc, occ_auc))
    print('Loss/uwb_total  Loss/uwb_pred  Loss/uwb_conf  uv_pred_auc  conf_auc  occlusion_auc')


if __name__ == '__main__':
    args = parse_args()
    evaluate(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        split=args.split,
        seq_len=args.seq_len,
        save_dir=args.save_dir,
    )
