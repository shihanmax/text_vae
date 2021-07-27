import os
import torch


class Config(object):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_text_path = os.path.join(base_dir, "./resource/imdb/review.txt")
    summary_path = os.path.join(base_dir, "./output/summary")
    model_path = os.path.join(base_dir, "./output/model")

    num_warmup_epochs = 3
    epochs = 10

    max_src_length = 110
    toy_size = 2000

    batch_size = 10
    lr = 1e-3
    gradient_clip = 10
    verbose = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    valid_ratio = 0.2
    test_ratio = 0.2

    # network config
    n_highway_layers = 3
    embedding_dim = 256
    encoder_hidden_dim = 256
    encoder_num_layers = 2
    z_dim = 128
    decoder_hidden_dim = 256
    decoder_num_layers = 1
    vocab_size = None
    max_decode_len = max_src_length
    
    # training
    not_early_stopping_at_first = 10
    es_with_no_improvement_after = 5

    sampling_text_list = [
        "this eerie build up works later on when coincidently",
        "the last shot  on to paris.",
        "her mission  to find life on earth.",
        "i love lots of things   star wars is one of my mains.",
        "it is everything it was supposed to be.",
    ]
