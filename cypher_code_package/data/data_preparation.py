from translate_package import (
    pd,
    train_test_split,
    Dataset,
    DataLoader,
    plt,
    torch,
    SequenceLengthBatchSampler,
    BucketSampler,
    partial,
    Union,
    Callable,
    ceil,
    np,
    TransformerSequences,
    nac,
    remove_mark_space,
    delete_guillemet_space
)

# sentence beginning with "Mooy li ko waral ci li ñu xamle waaye itam" is too long and must removed or corrected

# python translate_hyperparameter_tuning.py --model_generation "t5" --model_name "google-t5/t5-small" --tokenizer_name "sp" --use_bucketing --save_artifact

def augment(examples, src_label, p_word = 0.12554160436087158, p_char = 0.8269672653838092, max_words = 21):
    
    examples[src_label] = TransformerSequences(nac.RandomCharAug(action = 'swap', aug_word_p = p_word, aug_char_p = p_char, aug_word_max = max_words))(examples[src_label])[0]
    
    return examples

def augment_(examples, src_label, tgt_label):
    
    examples[src_label] = TransformerSequences(remove_mark_space, delete_guillemet_space)(examples[src_label])[0]
    
    examples[tgt_label] = TransformerSequences(remove_mark_space, delete_guillemet_space)(examples[tgt_label])[0]
    
    return examples

def tokenize(examples, tokenizer, src_label, tgt_label, model_generation):
    
    if model_generation in ["t5"]:
        
        eos_token = ""
        bos_token = ""
    
    else:
        
        eos_token = tokenizer._eos_token 
        bos_token = tokenizer._bos_token
    
    examples[src_label] = bos_token + examples[src_label] + eos_token

    examples[tgt_label] = bos_token + examples[tgt_label] + eos_token

    examples.update({key: value[0] for key, value in tokenizer(examples[src_label], return_tensors = 'pt').items()})

    examples.update({f'decoder_{key}': value[0] for key, value in tokenizer(examples[tgt_label], return_tensors = 'pt').items()})

    examples['labels'] = examples['decoder_input_ids']

    return examples

def apply_funcs(funcs, data):
    # Logic to apply the functions
    for func in funcs:
        data = func(data)
    return data

def sequences(examples, functions):

    for function in functions:
        
        examples = function(examples)
   
    return examples

class SentenceDataset(Dataset):
    
    def __init__(self, dataframe, transformers: Union[Callable, None] = None, source_column: str = 'WOLOF', target_column: str = 'FRENCH'):
        
        assert source_column in dataframe.columns.tolist() and target_column in dataframe.columns.tolist()
        
        self.source_sentences = dataframe[source_column].tolist()
        
        self.target_sentences = dataframe[target_column].tolist()
        
        self.transformers = transformers
        
        self.source_column = source_column
        
        self.target_column = target_column
    
    def __getitem__(self, index):
        
        source_sentence = self.source_sentences[index]
        
        target_sentence = self.target_sentences[index]
        
        sentences = {
            self.source_column: source_sentence,
            self.target_column: target_sentence
        }
        
        if not self.transformers is None:
            
            sentences = self.transformers(sentences)
        
        return sentences
    
    def __len__(self):
        
        return len(self.source_sentences)
        

def load_data(src_label, tgt_label, data_path, test_size, valid_size, seed):
    
    # load the dataset with pandas
    dataset_ = pd.read_csv(data_path)

    # split dataset between train, validation, and test sets
    if test_size == 1.0:

        dataset = {
            "test": partial(SentenceDataset, dataframe = dataset_, source_column = src_label, target_column = tgt_label),
        }

    else:

        train, test = train_test_split(
            dataset_, test_size=test_size + valid_size, random_state=seed
        )

        valid, test = train_test_split(
            test, test_size=test_size / (valid_size + test_size), random_state=seed
        )

        dataset = {
            "train": partial(SentenceDataset, dataframe = train, source_column = src_label, target_column = tgt_label),
            "val": partial(SentenceDataset, dataframe = valid, source_column = src_label, target_column = tgt_label),
            "test": partial(SentenceDataset, dataframe = test, source_column = src_label, target_column = tgt_label),
        }

    # The dataset actually contains 3 diff splits: train, validation, test.

    return dataset

def get_boundaries(dataset, sizes, min_count):

    length = []

    for i in range(len(dataset)):

        length.append(max(len(dataset[i]["input_ids"]), len(dataset[i]["labels"])))

    # Create histogram
    hist, bins, _ = plt.hist(length, bins=10)  # Adjust the number of bins as needed

    # Analyze the histogram
    # Identify peaks or gaps to determine the boundaries

    # Choose the boundaries based on the analysis
    boundaries = (
        [ceil(bins[0])]
        + [ceil(bin) for bin, count in zip(bins[1:], hist) if count > min_count]
        + [np.inf]
    )

    boundaries = boundaries[:-1]

    # define batch sizes and samplers
    batch_sizes = [
        sizes[i] if (i + 1) < len(sizes) else sizes[-1] for i in range(len(boundaries))
    ]

    return boundaries, batch_sizes

def collate_fn_trunc(batch, input_max_len, label_max_len, eos_token_id, pad_token_id, keys: list = ['input_ids', 'attention_mask', 'labels']):
    
    from torch.nn.utils.rnn import pad_sequence

    df_dict = {key: [] for key in keys}

    for b in batch:

        for key in df_dict:

            df_dict[key].append(b[key])

    padded_sequences = {}

    for key in df_dict:

        max_len = label_max_len if 'decoder' in key or 'label' in key else input_max_len

        padding_value = 0 if 'mask' in key else pad_token_id # must be take care

        # Pad the input sequences to have the same length
        padded_sequences[key] = pad_sequence(df_dict[key], batch_first=True, padding_value = padding_value)[:,:max_len]

        # eos token if it is not the case
        if not 'mask' in key:

             padded_sequences[key][:, -1:][(padded_sequences[key][:, -1:] != eos_token_id) & (padded_sequences[key][:, -1:] != pad_token_id)] = eos_token_id

    return padded_sequences

# define padding collate function
def pad_collate(batch, padding_value):

    X = [b["input_ids"] for b in batch]
    att = [b["attention_mask"] for b in batch]
    y = [b["labels"] for b in batch]

    X_ = torch.nn.utils.rnn.pad_sequence(
        X, batch_first=True, padding_value=padding_value
    )
    att_ = torch.nn.utils.rnn.pad_sequence(att, batch_first=True, padding_value=0)
    y_ = torch.nn.utils.rnn.pad_sequence(
        y, batch_first=True, padding_value=padding_value
    )

    return {"input_ids": X_, "attention_mask": att_, "labels": y_}


def get_loaders(
    tokenizer,
    model_generation,
    src_label,
    tgt_label,
    sizes,
    data_path,
    test_size,
    valid_size,
    seed,
    p_word,
    p_char,
    max_words,
    count,
    src_max_len,
    tgt_max_len,
    num_workers,
    device,
    use_bucketing,
    use_truncation,
    batch_size,
):

    # get dataset
    dataset = load_data(src_label, tgt_label, data_path, test_size, valid_size, seed)
    
    # analysis transformations
    
    a_transformers = partial(sequences, 
                functions = [
                    partial(augment_, src_label = src_label, tgt_label = tgt_label),
                    partial(tokenize, tokenizer = tokenizer, src_label = src_label, tgt_label = tgt_label, model_generation = model_generation)
            ])
    
    # training transformations
    t_transformers = partial(sequences, 
                functions = [
                    partial(augment, src_label = src_label, p_word = p_word, p_char = p_char, max_words = max_words),
                    partial(augment_, src_label = src_label, tgt_label = tgt_label),
                    partial(tokenize, tokenizer = tokenizer, src_label = src_label, tgt_label = tgt_label, model_generation = model_generation)
            ])
    
    if use_bucketing:
        
        if use_truncation:
            
            # initialize loaders
            train_sampler = BucketSampler(
                dataset["train"](transformers=a_transformers),
                batch_size=batch_size,
                input_key="input_ids",
                label_key="labels",
            )

            valid_sampler = BucketSampler(
                dataset["val"](transformers=a_transformers),
                batch_size=batch_size,
                input_key="input_ids",
                label_key="labels",
            )

            test_sampler = BucketSampler(
                dataset["test"](transformers=a_transformers),
                batch_size=batch_size,
                input_key="input_ids",
                label_key="labels",
            )
            
            # add transformations
            dataset = {s: dataset[s](transformers = t_transformers) for s in dataset}
            
            # define data loaders
            train_loader = DataLoader(
                dataset["train"],
                batch_sampler=train_sampler,
                collate_fn = partial(collate_fn_trunc, input_max_len = src_max_len, label_max_len = tgt_max_len,
                                    eos_token_id = tokenizer.eos_token_id, pad_token_id = tokenizer.pad_token_id),
                num_workers=num_workers,
                pin_memory=True if device in ["cuda", "gpu"] else False,
            )
            valid_loader = DataLoader(
                dataset["val"],
                batch_sampler=valid_sampler,
                collate_fn=partial(collate_fn_trunc, input_max_len = src_max_len, label_max_len = tgt_max_len,
                                    eos_token_id = tokenizer.eos_token_id, pad_token_id = tokenizer.pad_token_id),
                num_workers=num_workers,
                pin_memory=True if device in ["cuda", "gpu"] else False,
            )
            test_loader = DataLoader(
                dataset["test"],
                batch_sampler=test_sampler,
                collate_fn=partial(collate_fn_trunc, input_max_len = src_max_len, label_max_len = tgt_max_len,
                                    eos_token_id = tokenizer.eos_token_id, pad_token_id = tokenizer.pad_token_id),
                num_workers=num_workers,
                pin_memory=True if device in ["cuda", "gpu"] else False,
            )
        
        else:
            
            # get boundaries
            boundaries, batch_sizes = get_boundaries(dataset['train'](transformers = a_transformers), sizes, count)

            # initialize loaders
            train_sampler = SequenceLengthBatchSampler(
                dataset["train"](transformers=a_transformers),
                boundaries=boundaries,
                batch_sizes=batch_sizes,
                input_key="input_ids",
                label_key="labels",
            )

            valid_sampler = SequenceLengthBatchSampler(
                dataset["val"](transformers=a_transformers),
                boundaries=boundaries,
                batch_sizes=batch_sizes,
                input_key="input_ids",
                label_key="labels",
            )

            test_sampler = SequenceLengthBatchSampler(
                dataset["test"](transformers=a_transformers),
                boundaries=boundaries,
                batch_sizes=batch_sizes,
                input_key="input_ids",
                label_key="labels",
            )
            
            # add transformations
            dataset = {s: dataset[s](transformers = t_transformers) for s in dataset}
            
            # define data loaders
            train_loader = DataLoader(
                dataset["train"],
                batch_sampler=train_sampler,
                collate_fn=partial(pad_collate, padding_value=tokenizer.pad_token_id),
                num_workers=num_workers,
                pin_memory=True if device in ["cuda", "gpu"] else False,
            )
            valid_loader = DataLoader(
                dataset["val"],
                batch_sampler=valid_sampler,
                collate_fn=partial(pad_collate, padding_value=tokenizer.pad_token_id),
                num_workers=num_workers,
                pin_memory=True if device in ["cuda", "gpu"] else False,
            )
            test_loader = DataLoader(
                dataset["test"],
                batch_sampler=test_sampler,
                collate_fn=partial(pad_collate, padding_value=tokenizer.pad_token_id),
                num_workers=num_workers,
                pin_memory=True if device in ["cuda", "gpu"] else False,
            )

    else:
        
        # add transformations
        dataset = {s: dataset[s](transformers = t_transformers) for s in dataset}

        if "train" in dataset:
            # define data loaders
            train_loader = DataLoader(
                dataset["train"],
                batch_size=batch_size,
                collate_fn=partial(pad_collate, padding_value=tokenizer.pad_token_id),
                num_workers=num_workers,
                pin_memory=True if device in ["cuda", "gpu"] else False,
                shuffle=True,
            )

        if "val" in dataset:

            valid_loader = DataLoader(
                dataset["val"],
                batch_size=batch_size,
                collate_fn=partial(pad_collate, padding_value=tokenizer.pad_token_id),
                num_workers=num_workers,
                pin_memory=True if device in ["cuda", "gpu"] else False,
            )

        if "test" in dataset:

            test_loader = DataLoader(
                dataset["test"],
                batch_size=batch_size,
                collate_fn=partial(pad_collate, padding_value=tokenizer.pad_token_id),
                num_workers=num_workers,
                pin_memory=True if device in ["cuda", "gpu"] else False,
            )

    if "train" in dataset and "val" in dataset:

        return {
            "train_loader": train_loader,
            "valid_loader": valid_loader,
            "test_loader": test_loader,
        }

    else:

        return {"test_loader": test_loader}
