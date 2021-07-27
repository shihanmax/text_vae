import torch


class Vocab(object):
    def __init__(
        self, str2idx, idx2str, pad_idx, unk_idx, sos_idx, eos_idx, pad, unk,
        sos, eos,
    ):
        self.str2idx = str2idx
        self.idx2str = idx2str
        self.pad = pad
        self.unk = unk
        self.sos = sos
        self.eos = eos
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

        self.size = len(str2idx)


def translate_text(text, vocab: Vocab, max_length, tokenize):
    if isinstance(text, str):
        indices = [
            vocab.str2idx.get(token, vocab.unk_idx) for token in tokenize(text)
        ]
        indices = indices[:max_length]
        
        valid_length = len(indices)
        if valid_length < max_length:
            indices.extend([vocab.pad_idx] * (max_length - valid_length))
        return indices, valid_length

    else:
        raise TypeError(f"Unsupported type:{type(text)}")


def translate_idx(indices, vocab, drop_pad=True):
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().tolist()

    if isinstance(indices[0], int):
        tokens = [vocab.idx2str.get(token_idx) for token_idx in indices]
        while tokens and tokens[-1] == vocab.pad:
            tokens.pop()
        return tokens
    
    elif isinstance(indices[0], list):
        return [translate_idx(i, vocab) for i in indices]
    else:
        raise TypeError(f"Unsupported type:{type(indices)}")
