import os
import torch 
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils.dist_utils import all_gather

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level, pos_label=1.):
    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

def do_test_target_with_prototypes(args, models, prototypes, target_loader, known_classes=None):
    if known_classes is None:
        known_classes = args.known_classes

    feature_extractor = models["feature_extractor"]
    cls_rel = models["cls_rel"]

    correct_predictions = 0
    correct_unk_predictions = 0

    closed_set_predicted_category = {}
    normality_scores = {}
    ent_normality_scores = {}
    gt_labels = {}
    binary_feats = {}
    binary_labels = {}

    # perform eval
    with torch.no_grad():
        unknown = 0
        known = 0

        for it, (data_t, class_l_t, index) in enumerate(tqdm(target_loader)):
            data_t, class_l_t = data_t.to(args.device), class_l_t.to(args.device)
            data_t_feat = feature_extractor.forward(data_t)

            batch_pos = 0
            batch = np.zeros((args.batch_size,feature_extractor.output_num()),dtype=np.float32)
            categories_in_batch = []
            predictions_per_category = torch.zeros((known_classes,2))
            attns_per_category = torch.zeros((known_classes,args.transf_depth,args.transf_n_heads,3,3))
            category = 0

            while category < known_classes:
                cat_prototype = prototypes[category]
                batch[batch_pos] = cat_prototype
                categories_in_batch.append(category)
                batch_pos += 1

                if batch_pos == args.batch_size or category + 1 == known_classes:
                    # batch is full
                    torch_batch = torch.from_numpy(batch[:batch_pos]).to(args.device)
                    feats = data_t_feat.expand(batch_pos,-1)

                    aggregated_batch = torch.cat((feats, torch_batch), 1)

                    out, _, _ = cls_rel(aggregated_batch, return_attention=True, return_feats=True)
                    # check if the size per sample is 2 (cross-entropy) or 1 (L2)
                    if out.shape[1] <= 1:                        
                        out = - out
                        #out = 1 - torch.sigmoid(out)
                    relation_l = torch.ones((batch_pos),dtype=torch.long,device=aggregated_batch.device)
                    relation_l[torch.tensor(categories_in_batch,device=class_l_t.device)==class_l_t] = 0

                    out = out.cpu()
                    predictions_per_category[categories_in_batch] = out
                    batch_pos = 0
                    categories_in_batch = []
                category += 1
            
            class_pred_same_category = predictions_per_category[:,0] 
            class_pred_same_category = nn.Softmax(0)(class_pred_same_category).numpy()

            # in order to understand if this sample belongs to a known class we need to check predictions 
            # for all classes. We look for maximum prediction. The category corresponding to maximum prediction 
            # is the most probable. However if the probability for this max predictions is still lower than a certain thres it means 
            # probably that this is an unknown sample

            max_pred_value = np.max(class_pred_same_category)
            max_pred_category = np.argmax(class_pred_same_category)

            closed_set_predicted_category[index.item()] = max_pred_category
            normality_scores[index.item()] = max_pred_value
            gt_labels[index.item()] = class_l_t.item()

        target_scores = {}
        target_gt_labels = {}
        target_predicted_labels = {}
        
        if args.distributed:
            all_scores = all_gather(normality_scores)
            all_labels = all_gather(gt_labels)
            all_predictions = all_gather(closed_set_predicted_category)
            
            for dic in all_scores:
                target_scores.update(dic)
            for dic in all_labels:
                target_gt_labels.update(dic)
            for dic in all_predictions:
                target_predicted_labels.update(dic)
        else:
            target_scores = normality_scores
            target_ent_scores = ent_normality_scores
            target_gt_labels = gt_labels
            target_predicted_labels = closed_set_predicted_category

        normality_scores = torch.tensor([target_scores[k] for k in sorted(target_scores.keys())])
        gt_labels = torch.tensor([target_gt_labels[k] for k in sorted(target_gt_labels.keys())])

        # compute metrics 
        normality_scores = np.array(normality_scores)

        ood_labels = np.zeros_like(normality_scores)
        ood_labels[gt_labels < known_classes] = 1

        scores = np.array(normality_scores)
        auroc = roc_auc_score(ood_labels, scores)
        
        recall_level = 0.95
        fpr_auroc = fpr_and_fdr_at_recall(ood_labels, scores, recall_level)

        print("Auroc %f" % (auroc))
        print("FPR95 %f" % (fpr_auroc))

        return auroc

@torch.no_grad()
def do_knn_distance_eval(args, models, source_loader, target_loader, known_classes=None):
    if known_classes is None:
        known_classes = args.known_classes

    # we need to compare each test sample with all training samples
    feature_extractor = models["feature_extractor"]
    cls_rel = models["cls_rel"]

    device = args.device

    assert target_loader.batch_size == 1, "Target loader batch size should be 1"

    predictions = torch.zeros((len(target_loader.dataset), len(source_loader.dataset)))
    gt_labels = torch.zeros((len(target_loader.dataset)), dtype=torch.long)

    for test_idx, test_batch in enumerate(tqdm(target_loader)):
        images, labels, indices = test_batch 
        gt_labels[indices] = labels

        images = images.to(device)
        test_feats = feature_extractor.forward(images)

        for train_batch in source_loader:
            train_images, train_labels, train_ids = train_batch 

            train_images = train_images.to(device) 
            train_feats = feature_extractor.forward(train_images)

            aggregated_batch = torch.cat((test_feats.expand(train_feats.shape[0], -1), train_feats), 1)
            out = - cls_rel(aggregated_batch)
            predictions[indices, train_ids] = out.squeeze().cpu()

    if args.distributed:
        all_labels = all_gather(gt_labels)
        all_preds = all_gather(predictions)

        predictions_accum = all_preds[0]
        labels_accum = all_labels[0]

        for idx in range(1,len(all_labels)):
            predictions_accum += all_preds[idx]
            labels_accum += all_labels[idx]

        predictions = predictions_accum
        gt_labels = labels_accum

    K = args.NNK
    values, indices = predictions.topk(k=K, dim=1)
    normality_scores = values.mean(dim=1).numpy()

    # compute metrics 
    ood_labels = np.zeros_like(normality_scores)
    ood_labels[gt_labels < known_classes] = 1

    scores = np.array(normality_scores)
    auroc = roc_auc_score(ood_labels, scores)
    
    recall_level = 0.95
    fpr_auroc = fpr_and_fdr_at_recall(ood_labels, scores, recall_level)

    print("Auroc %f" % (auroc))
    print("FPR95 %f" % (fpr_auroc))

    return auroc



