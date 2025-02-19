import numpy as np

########## classification metrics ##########
def get_TP_TN_FP_FN(predicted_indices, target_indices, num_classes):
    # Convert indices to one-hot encoding
    predicted_onehot = np.eye(num_classes)[predicted_indices.astype(int)]
    target_onehot = np.eye(num_classes)[target_indices.astype(int)]
    predicted = predicted_onehot.reshape(-1, num_classes)
    target = target_onehot.reshape(-1, num_classes)
    
    TP, TN, FP, FN = np.zeros(num_classes), np.zeros(num_classes), np.zeros(num_classes), np.zeros(num_classes)

    for i in range(num_classes):
        TP[i] = np.sum((predicted[:, i] == 1) & (target[:, i] == 1))
        TN[i] = np.sum((predicted[:, i] == 0) & (target[:, i] == 0))
        FP[i] = np.sum((predicted[:, i] == 1) & (target[:, i] == 0))
        FN[i] = np.sum((predicted[:, i] == 0) & (target[:, i] == 1))
    return TP, TN, FP, FN

def get_metrics(TP, TN, FP, FN):
    # Calculate the metrics for each class and overall (Macro-averaging and Micro-averaging)
    accuracy_single = (TP + TN) / (TP + TN + FP + FN)
    accuracy_micro_avg = (np.sum(TP) + np.sum(TN)) / (np.sum(TP) + np.sum(TN) + np.sum(FP) + np.sum(FN))
    accuracy_macro_avg = np.nanmean(accuracy_single)
    accuracy = {'single_accuracy': accuracy_single, 'micro_avg_accuracy': accuracy_micro_avg, 'macro_avg_accuracy': accuracy_macro_avg}
    
    # Calculate precision with division handling
    with np.errstate(divide='ignore', invalid='ignore'):
        precision_single = TP / (TP + FP)
    precision_macro_avg = np.nanmean(precision_single)
    precision = {'single_precision': precision_single, 'macro_avg_precision': precision_macro_avg}
    
    # Calculate recall with division handling
    with np.errstate(divide='ignore', invalid='ignore'):
        recall_single = TP / (TP + FN)
    recall_macro_avg = np.nanmean(recall_single)
    recall = {'single_recall': recall_single, 'macro_avg_recall': recall_macro_avg}
    
    # Calculate F1 score with division handling
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_single = (2 * precision_single * recall_single) / (precision_single + recall_single)
    f1_macro_avg = np.nanmean(f1_single)
    f1 = {'single_f1': f1_single, 'macro_avg_f1': f1_macro_avg}
    
    return accuracy, precision, recall, f1



########## bounding box metrics ##########ù
def get_iou_1d(interval1, interval2):
    start1, end1 = interval1
    start2, end2 = interval2

    # Compute intersection
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    inter_length = max(0, inter_end - inter_start)  # If negative, no intersection

    # Compute union
    total_length = (end1 - start1) + (end2 - start2) - inter_length

    return inter_length / total_length if total_length > 0 else 0

import numpy as np

def get_windows_TP_FP_FN(predicted_intervals, target_intervals, iou_threshold=0.5):
    TP, FP, FN = 0, 0, 0

    if len(predicted_intervals) == 0:
        FN = len(target_intervals)
        return TP, FP, FN
    
    if len(target_intervals) == 0:
        FP = len(predicted_intervals)
        return TP, FP, FN
    
    iou_matrix = np.zeros((len(target_intervals), len(predicted_intervals)))

    # Compute IoU for each pair
    for i, target_interval in enumerate(target_intervals):
        for j, predicted_interval in enumerate(predicted_intervals):
            iou_matrix[i, j] = get_iou_1d(target_interval, predicted_interval)

    # Find the best IoU for each target interval
    max_iou = np.max(iou_matrix, axis=1)
    
    # Count TP, FP, FN
    TP = np.sum(max_iou >= iou_threshold)
    FP = len(predicted_intervals) - TP
    FN = len(target_intervals) - TP

    return TP, FP, FN

def get_windows_metrics(TP, FP, FN):
    # Calculate the metrics for each class and overall
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
