import json
import numpy as np
import os
from tqdm import tqdm
from preprocessData import calculateMetrics

def calculateAveragePrecision(probs, gt_indices):
    """
    Calculates Average Precision (AP) for a single sample.
    probs: list of probabilities for each segment
    gt_indices: list of ground truth indices
    """
    if not gt_indices:
        return 0.0
        
    # Sort indices by probability (descending)
    ranked_indices = np.argsort(probs)[::-1]
    gt_set = set(gt_indices)
    
    score = 0.0
    num_hits = 0.0
    
    for i, idx in enumerate(ranked_indices):
        if idx in gt_set:
            num_hits += 1.0
            precision_at_i = num_hits / (i + 1.0)
            score += precision_at_i
            
    return score / len(gt_set)

def main():
    results_file = "prediction_results_linear.json"
    
    if not os.path.exists(results_file):
        print(f"File {results_file} not found.")
        return

    with open(results_file, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {results_file}")

    # Calculate Mean Average Precision (mAP)
    print("Calculating Mean Average Precision (mAP)...")
    ap_scores = []
    for sample in data:
        probs = sample['probs']
        gt_indices = sample['gt_indices']
        ap = calculateAveragePrecision(probs, gt_indices)
        ap_scores.append(ap)
    
    mean_ap = np.mean(ap_scores)
    print(f"Mean Average Precision (mAP): {mean_ap:.4f}")

    best_threshold = 0.0
    best_f1 = -1.0
    best_metrics = {}
    best_num_preds = []

    # Search range: 0.01 to 0.99 with step 0.01
    thresholds = np.arange(0.01, 1.00, 0.01)
    
    print("Searching for best threshold...")
    
    for threshold in tqdm(thresholds):
        current_metrics = {
            'hit_rate': [],
            'precision': [],
            'recall': [],
            'f1': [],
        }
        current_num_preds = []

        for sample in data:
            probs = sample['probs']
            gt_indices = sample['gt_indices']
            
            # Get predicted indices based on current threshold
            pred_indices = [i for i, p in enumerate(probs) if p > threshold]
            current_num_preds.append(len(pred_indices))
            
            # Calculate metrics for this sample
            metrics = calculateMetrics(pred_indices, gt_indices)
            
            for k, v in metrics.items():
                current_metrics[k].append(v)
        
        # Calculate average F1 for this threshold
        avg_f1 = np.mean(current_metrics['f1'])
        
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_threshold = threshold
            # Store all average metrics for the best threshold
            best_metrics = {k: np.mean(v) for k, v in current_metrics.items()}
            best_num_preds = current_num_preds

    print("\n" + "="*30)
    print(f"Best Threshold Found: {best_threshold:.2f}")
    print("="*30)
    print(f"F1 Score:  {best_metrics['f1']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall:    {best_metrics['recall']:.4f}")
    print(f"Hit Rate:  {best_metrics['hit_rate']:.4f}")
    print(f"mAP:       {mean_ap:.4f}")
    print(f"Avg Preds: {np.mean(best_num_preds):.2f}({np.std(best_num_preds):.2f})")
    print("="*30)

if __name__ == "__main__":
    main()
