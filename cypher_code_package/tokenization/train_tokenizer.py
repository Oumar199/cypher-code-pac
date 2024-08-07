
from translate_package import argparse, spm, pd, os, Tokenizer, models, pre_tokenizers, trainers, processors, decoders
from translate_package.errors import TokenizerException

def train_tokenizer(arguments):
    
    # recuperate dataset
    dataset = pd.read_csv(arguments.dataset_file)
    
    if arguments.name == 'bpe':
        
        # initialize tokenizer
        tokenizer = Tokenizer(models.BPE())
        
        # initialize pre-to
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False)
        
        # initialize trainer
        trainer = trainers.BpeTrainer(vocab_size = arguments.vocab_size, special_tokens = ['<s>', '<pad>', '</s>', '<unk>'])
        
        # iterate over the dataset and return the sentences
        def get_training_corpus():
            
            sentences = dataset[arguments.src_label].tolist() + dataset[arguments.tgt_label].tolist()
            
            for i in range(0, len(sentences), arguments.batch_size):
                
                yield sentences[i: i + arguments.batch_size]
                
        # train the tokenizer
        tokenizer.train_from_iterator(get_training_corpus(), trainer = trainer)
        
        tokenizer.post_processor = processors.ByteLevel(trim_offsets = False)
        
        tokenizer.decoder = decoders.ByteLevel()
        
        # get path
        tk_path = os.path.join(arguments.save_path, f'{arguments.file_name}.json')
        
        tokenizer.save(tk_path)
        
        print(f"The Byte Pair Encoding tokenizer was saved as {tk_path}!")
    
    elif arguments.name == 'sp':
        
        # print sentences into a file
        with open('sents.txt', 'w', encoding = 'utf-8') as f:
            
            sentences =  dataset[arguments.src_label].tolist() + dataset[arguments.tgt_label].tolist()
            
            for i in range(0, len(sentences), arguments.batch_size):
                
                sents = sentences[i: i + arguments.batch_size]
                
                f.write("\n".join(sents)+'\n')
        
        # get path
        tk_path = os.path.join(arguments.save_path, arguments.file_name)
              
        # initialize sentence piece trainer
        spm.SentencePieceTrainer.Train(input = f'sents.txt',
                               model_prefix=os.path.join(arguments.save_path, arguments.file_name),
                               vocab_size=arguments.vocab_size,
                               character_coverage=1.0,
                               pad_id=0,                
                               eos_id=1,
                               unk_id=2,
                               bos_id=3,
                               pad_piece='<pad>',
                               eos_piece='</s>',
                               unk_piece='<unk>',
                               bos_piece='<s>',
                              )
        
        # remove file
        os.remove('sents.txt')
        
        print(f"The Sentence Piece tokenizer was saved as {tk_path}(.model / for model)!")
        
    else:
        
        raise TokenizerException("You can only train a sentence piece (as 'sp') tokenizer, or a byte pair encoding (as 'bpe') tokenizer!")
    