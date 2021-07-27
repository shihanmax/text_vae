import random
import logging
import torch

from tqdm import tqdm
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class RawTextProvider(object):
    
    def __init__(self):
        pass
    
    def load(self, raw_data_path, valid_ratio, test_ratio, toy_size):
        all_text = []
        
        with open(raw_data_path) as frd:
            for cnt, line in enumerate(tqdm(frd.readlines())):
                all_text.append(word_tokenize(line))
                if toy_size and cnt == toy_size:
                    break
                
        if toy_size:
            logger.info(f"debugging pipeline with {toy_size} pieces of data")
            
        logger.info("Done loading and tokenize source files")
        
        random.seed(1)
        random.shuffle(all_text)
        
        total_cnt = len(all_text)
        valid_cnt = int(total_cnt * valid_ratio)
        test_cnt = int(total_cnt * test_ratio)
        
        valid_text = all_text[:valid_cnt]
        test_text = all_text[-test_cnt:]
        train_text = all_text[valid_cnt: -test_cnt]
        
        return train_text, valid_text, test_text


class VAEDataset(Dataset):
    
    def __init__(self, all_text, max_src_length, vocab):
        self.all_text = all_text
        self.str2idx = vocab.str2idx
        self.max_src_length = max_src_length
        
        self.sos_idx = vocab.sos_idx
        self.eos_idx = vocab.eos_idx
        self.pad_idx = vocab.pad_idx
        self.unk_idx = vocab.unk_idx
    
    def build_single(self, text):
        token_idx = [
            self.str2idx.get(token, self.unk_idx) for token in text
        ]
        token_idx.insert(0, self.sos_idx)  # add <sos>
        
        curr_len = len(token_idx)
        
        if curr_len > self.max_src_length - 1:
            token_idx = token_idx[:self.max_src_length - 1]
        
        token_idx.append(self.eos_idx)  # add <eos>
        
        curr_len = len(token_idx)
        valid_length = curr_len
        
        if curr_len < self.max_src_length:
            pad_len = self.max_src_length - curr_len
            token_idx.extend([self.pad_idx] * pad_len)
        
        return {
            "x": torch.tensor(token_idx, dtype=torch.long),
            "valid_length": torch.tensor(valid_length, dtype=torch.long)
        } 
    
    def __getitem__(self, item):
        return self.build_single(self.all_text[item])
    
    def __len__(self):
        return len(self.all_text)
