import torch
import numpy as np

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}

class LPRLabelEncoder:
    def __init__(self, chars=CHARS):
        self.chars = chars
        self.char_dict = {char: i for i, char in enumerate(self.chars)}
        self.idx_dict = {i: char for i, char in enumerate(self.chars)}

    def encode(self, text):
        return [self.char_dict[c] for c in text if c in self.char_dict]

    def decode(self, preds):
        """
        Greedy decode for CTC
        preds: (T, N, C) or (N, C) tensor or numpy array
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
            
        pred_labels = []
        # Support batch decoding
        if len(preds.shape) == 3: 
             pass 
        
        # Simple greedy decode logic for single sequence of indices (after argmax)
        pass 

    def decode_greedy(self, logits):
        # logits: (N, C, T)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1) # (N, T)
        
        results = []
        for i in range(preds.shape[0]):
            pred = preds[i]
            res = ""
            prev = -1
            for val in pred:
                if val != prev and val != len(self.chars): 
                    res += self.idx_dict.get(val, "")
                prev = val
            results.append(res)
        return results

    def decode_beam(self, logits, beam_width=10):
        # logits: (N, C, T) or (N, T, C) - LPRNet usually returns (N, C, T)
        # We need (N, T, C) for easier time-step iteration
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        
        # Check shape, if (N, C, T) -> transpose to (N, T, C)
        if logits.shape[1] == len(self.chars) + 1:
            logits = np.transpose(logits, (0, 2, 1))

        probs = logits 
        
        n_batch, n_time, n_class = probs.shape
        blank_idx = len(self.chars)
        results = []

        for b in range(n_batch):
            # Initialize with empty beam: (score, text_indices, last_char_idx)
            # using log probability 0.0 for start
            beams = [(0.0, [], -1)] 
            
            for t in range(n_time):
                next_beams = []
                step_probs = probs[b, t] # (C,)
                
                # Optimization: only consider top k candidates at each step to speed up
                # standard beam search expands all, but we limit to beam_width
                top_k_indices = np.argsort(step_probs)[-beam_width:]
                
                for prefix_score, prefix, last_char in beams:
                    for c in top_k_indices:
                        p = step_probs[c]
                        
                        pattern_score = prefix_score + p
                        
                        # CTC logic
                        new_prefix = list(prefix)
                        new_last_char = c
                        
                        if c == blank_idx:
                            next_beams.append((pattern_score, new_prefix, c))
                        else:
                            if c != last_char:
                                new_prefix.append(c)
                           
                            next_beams.append((pattern_score, new_prefix, c))
                
                # Sort and trim
                next_beams.sort(key=lambda x: x[0], reverse=True)
                beams = next_beams[:beam_width]

            # Decode best beam
            best_beam = beams[0]
            best_path_indices = best_beam[1] # raw indices sequence including blanks and repeats
            
            # Collapse CTC
            res = ""
            prev = -1
            for val in best_path_indices:
                 if val != blank_idx: 
                    pass
            
            # Re-run correctly:
            final_indices = beams[0][1] 
            res = ""
            prev = -1
            
            for idx in beams[0][1]:
                res += self.idx_dict.get(idx, "")
            results.append(res)
            
        return results

from src.ocr.lprnet import build_lprnet

def get_model(class_num=len(CHARS)+1):
    # class_num including blank
    return build_lprnet(class_num=class_num)
