import sys
sys.path.append("..")

import torch
import logging
from torch.utils.data import DataLoader
from torch.optim import AdamW
from nltk.tokenize import word_tokenize
from nlkit.word2vec import build_vocab_from_text_file
from nlkit.utils import weight_init, get_linear_schedule_with_warmup_ep

from vae_text_generation.utils import Vocab
from vae_text_generation.model import VAE
from vae_text_generation.trainer import Trainer
from vae_text_generation.data import VAEDataset, RawTextProvider
from vae_text_generation.config import Config

logging.basicConfig(level=logging.INFO)

str2idx, idx2str = build_vocab_from_text_file(
    file_path=Config.raw_text_path, freq_threshold=10, tokenizer=word_tokenize,
    keep_tokens=("<pad>", "<unk>", "<sos>", "<eos>"),
)
# <pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3
vocab = Vocab(str2idx, idx2str, 0, 1, 2, 3, "<pad>", "<unk>", "<sos>", "<eos>")

Config.vocab_size = vocab.size

vae = VAE(
    n_highway_layers=Config.n_highway_layers, 
    embedding_dim=Config.embedding_dim, 
    encoder_hidden_dim=Config.encoder_hidden_dim, 
    encoder_num_layers=Config.encoder_num_layers, 
    z_dim=Config.z_dim,
    decoder_hidden_dim=Config.decoder_hidden_dim, 
    decoder_num_layers=Config.decoder_num_layers, 
    vocab_size=Config.vocab_size, 
    max_decode_len=Config.max_decode_len,
)

optimizer = AdamW(
    vae.parameters(), lr=Config.lr, 
)

lr_scheduler = get_linear_schedule_with_warmup_ep(
    optimizer=optimizer, 
    num_warmup_epochs=Config.num_warmup_epochs, 
    total_epochs=Config.epochs, 
)

raw_text_provider = RawTextProvider()
train_text, valid_text, test_text = raw_text_provider.load(
    Config.raw_text_path, valid_ratio=Config.valid_ratio, 
    test_ratio=Config.test_ratio, toy_size=Config.toy_size,
)

train_dataset = VAEDataset(
    train_text, 
    max_src_length=Config.max_src_length, 
    vocab=vocab,
)

valid_dataset = VAEDataset(
    valid_text, 
    max_src_length=Config.max_src_length, 
    vocab=vocab,
)

test_dataset = VAEDataset(
    test_text,
    max_src_length=Config.max_src_length, 
    vocab=vocab,
)

train_data_loader = DataLoader(train_dataset, batch_size=Config.batch_size)
valid_data_loader = DataLoader(valid_dataset, batch_size=Config.batch_size)
test_data_loader = DataLoader(test_dataset, batch_size=Config.batch_size)

trainer = Trainer(
    model=vae, 
    train_data_loader=train_data_loader, 
    valid_data_loader=valid_data_loader, 
    test_data_loader=test_data_loader, 
    lr_scheduler=lr_scheduler, 
    optimizer=optimizer, 
    weight_init=weight_init, 
    summary_path=Config.summary_path, 
    device=Config.device, 
    criterion=None,
    total_epoch=Config.epochs, 
    model_path=Config.model_path,
    gradient_clip=Config.gradient_clip,
    verbose=Config.verbose,
    not_early_stopping_at_first=Config.not_early_stopping_at_first,
    es_with_no_improvement_after=Config.es_with_no_improvement_after,
    sampling_text_list=Config.sampling_text_list,
    vocab=vocab,
)

trainer.start_train()
