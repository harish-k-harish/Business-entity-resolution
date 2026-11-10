import os
import pickle
import numpy as np
import lightgbm as lgb
try:
    from .features import FEATURE_NAMES
except ImportError:
    from features import FEATURE_NAMES

def train_matcher_model(X_train: np.ndarray, y_train: np.ndarray, model_save_path: str = None) -> lgb.LGBMClassifier:
    """Train a tuned LightGBM binary classifier on pairwise similarity features."""
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    clf.fit(X_train, y_train)
    
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        with open(model_save_path, 'wb') as f:
            pickle.dump(clf, f)
            
    return clf

def load_matcher_model(model_path: str) -> lgb.LGBMClassifier:
    """Load a pretrained LightGBM classifier from disk."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def optimize_threshold_macro_f05(val_pairs: list, val_probs: np.ndarray, gt_dict: dict, eval_s1_ids: list):
    """Grid search for the probability threshold that maximizes Macro F_0.5 on validation data."""
    best_thresh = 0.68
    best_score = 0.0
    
    # Pre-group predictions by s1_id for rapid threshold scanning
    pairs_by_s1 = {}
    for (s1_id, tid), prob in zip(val_pairs, val_probs):
        if s1_id not in pairs_by_s1:
            pairs_by_s1[s1_id] = []
        pairs_by_s1[s1_id].append((tid, prob))
        
    for thresh in np.arange(0.50, 0.92, 0.02):
        scores = []
        for s1_id in eval_s1_ids:
            cand_probs = pairs_by_s1.get(s1_id, [])
            preds = [tid for tid, p in cand_probs if p >= thresh]
            truth = gt_dict.get(s1_id, set())
            
            # Singleton handling
            if len(truth) == 0:
                scores.append(1.0 if len(preds) == 0 else 0.0)
            else:
                if len(preds) == 0:
                    scores.append(0.0)
                else:
                    tp = len(set(preds) & truth)
                    if tp == 0:
                        scores.append(0.0)
                    else:
                        prec = tp / len(preds)
                        rec = tp / len(truth)
                        f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
                        scores.append(f05)
                        
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_thresh = thresh
            
    return best_thresh, best_score
