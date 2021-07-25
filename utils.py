from nltk.tokenize import word_tokenize
from nlkit.word2vec import build_vocab_from_text_file


str_to_idx, idx_to_str = build_vocab_from_text_file(
    file_path="", freq_threshold=10, tokenizer=word_tokenize,
)
