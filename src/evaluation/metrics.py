import numpy as np

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def compute_ocr_metrics(preds, targets):
    """
    Computes OCR metrics.
    preds: List of predicted strings.
    targets: List of ground truth strings.
    
    Returns:
        dict: {
            'seq_acc': Sequence Level/Greedy Accuracy,
            'char_acc': Character Level Accuracy (1 - CER),
            'avg_edit_dist': Average Levenshtein Distance
        }
    """
    total_seq = len(preds)
    correct_seq = 0
    total_edit_dist = 0
    total_chars = 0
    total_correct_chars = 0 # Approximate based on edit distance
    
    for pred, target in zip(preds, targets):
        if pred == target:
            correct_seq += 1
            
        dist = levenshtein(pred, target)
        total_edit_dist += dist
        
        total_chars += len(target)
        
    seq_acc = correct_seq / total_seq if total_seq > 0 else 0
    avg_edit_dist = total_edit_dist / total_seq if total_seq > 0 else 0
    
    # Character Error Rate (CER) = Total Edit Distance / Total Characters
    cer = total_edit_dist / total_chars if total_chars > 0 else 0
    char_acc = 1 - cer
    
    return {
        "seq_acc": seq_acc,
        "char_acc": char_acc,
        "avg_edit_dist": avg_edit_dist
    }
