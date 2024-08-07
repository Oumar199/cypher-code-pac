from transformers import BartTokenizerFast
from transformers import T5TokenizerFast
import os

def load_tokenizer(tokenizer_name, model, dir_path, file_name):
    
    if tokenizer_name == "bpe":
        
        tokenizer_path = os.path.join(dir_path, f"{file_name}.json")
        
        if model == "bart":
        
            tokenizer = BartTokenizerFast(tokenizer_file=tokenizer_path)
        
        print(f"The Byte Pair Encoding tokenizer was successfully uploaded from {tokenizer_path}")
        
    elif tokenizer_name == "sp":
        
        tokenizer_path = os.path.join(dir_path, f"{file_name}.model")
        
        if model == 't5':
            
            tokenizer = T5TokenizerFast(vocab_file=tokenizer_path)
        
        print(f"The Sentence Piece tokenizer was successfully uploaded from {tokenizer_path}")
    
    return tokenizer