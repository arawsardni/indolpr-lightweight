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
        if len(preds.shape) == 3: # (T, N, C) typical for CTC output from pytorch (Time, Batch, Class) or (N, C, T) from LPRNet
             # LPRNet returns (N, C, T) -> need transpose to (T, N, C) for standard CTC usually, or (N, T) indices
             # Let's assume input is logits
             pass 
        
        # Simple greedy decode logic for single sequence of indices (after argmax)
        pass 

    def decode_greedy(self, logits):
        # logits: (N, C, T) - typical LPRNet output
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1) # (N, T)
        
        results = []
        for i in range(preds.shape[0]):
            pred = preds[i]
            res = ""
            prev = -1
            for val in pred:
                if val != prev and val != len(self.chars): # len(self.chars) is usually blank index in LPRNet implementation depends on config
                    res += self.idx_dict.get(val, "")
                prev = val
            results.append(res)
        return results

from src.ocr.lprnet import build_lprnet

def get_model(class_num=len(CHARS)+1):
    # class_num including blank
    return build_lprnet(class_num=class_num)
