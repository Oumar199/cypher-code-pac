from cypher_code_package import (
    pd,
    train_test_split,
    Dataset,
    DatasetDict,
    DataLoader,
    CodeGenTokenizerFast,
    plt,
    torch,
    SequenceLengthBatchSampler,
    partial,
    ceil,
    np,
)


def load_data(model_name, data_dir, test_size, valid_size, seed):

    # load the legal dataset with pandas
    legal_cypher = pd.read_csv(data_dir)

    # print head
    legal_cypher.head()

    # replace apostroph
    legal_cypher["contenu"] = legal_cypher["contenu"].map(lambda x: x.replace("’", "'"))

    # split dataset between train, validation, and test sets
    train, test = train_test_split(legal_cypher, test_size=test_size + valid_size, random_state=seed)

    valid, test = train_test_split(test, test_size=test_size / (valid_size + test_size), random_state=seed)

    dataset = {
        "train": Dataset.from_dict(
            {"label": train["cypher"], "text": train["contenu"]}
        ),
        "val": Dataset.from_dict({"label": valid["cypher"], "text": valid["contenu"]}),
        "test": Dataset.from_dict({"label": test["cypher"], "text": test["contenu"]}),
    }

    dataset = DatasetDict(dataset)

    tokenizer = get_tokenizer(model_name)

    # define tokenize function
    def tokenize_function(example, tokenizer):

        text_start = "Legal text ::\n\n"
        code_start = "\n\nCode Cypher ::\n\n"
        prompt = text_start + example["text"] + code_start + example["label"]
        tokens = tokenizer(prompt, return_tensors="pt")
        example["input_ids"] = tokens.input_ids[0]
        example["attention_mask"] = tokens.attention_mask[0]
        example["labels"] = example["input_ids"]

        return example

    # The dataset actually contains 3 diff splits: train, validation, test.
    # The tokenize_function code is handling all data across all splits in batches.
    tokenized_datasets = dataset.map(
        tokenize_function, fn_kwargs={"tokenizer": tokenizer}
    )

    print(f"Shapes of the datasets:")
    print(tokenized_datasets)

    return tokenized_datasets


def get_tokenizer(model_name):

    tokenizer = CodeGenTokenizerFast.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_boundaries(dataset, tokenizer, sizes, min_count):

    length = []

    for i in range(len(dataset["train"])):

        text_start = "Legal text ::\n\n"
        code_start = "\n\nCode Cypher ::\n\n"
        prompt = (
            text_start
            + dataset["train"][i]["text"]
            + code_start
            + dataset["train"][i]["label"]
        )

        tokens = tokenizer(prompt, return_tensors="pt").input_ids[0]

        length.append(len(tokens))

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


# define padding collate function
def pad_collate(batch, padding_value):

    X = [torch.tensor(b["input_ids"]) for b in batch]
    att = [torch.tensor(b["attention_mask"]) for b in batch]
    y = [torch.tensor(b["labels"]) for b in batch]

    X_ = torch.nn.utils.rnn.pad_sequence(
        X, batch_first=True, padding_value=padding_value
    )
    att_ = torch.nn.utils.rnn.pad_sequence(att, batch_first=True, padding_value=0)
    y_ = torch.nn.utils.rnn.pad_sequence(
        y, batch_first=True, padding_value=padding_value
    )

    return {"input_ids": X_, "attention_mask": att_, "labels": y_}


def get_loaders(
    model_name,
    sizes,
    data_dir,
    test_size,
    valid_size,
    seed,
    count,
    num_workers,
    device,
    use_bucketing,
    batch_size,
):

    tokenizer = get_tokenizer(model_name)

    # get dataset
    dataset = load_data(model_name, data_dir, test_size, valid_size, seed)

    if use_bucketing:
        # get boundaries
        boundaries, batch_sizes = get_boundaries(dataset, tokenizer, sizes, count)

        # remove unnecessary columns
        dataset = dataset.remove_columns(["text", "label"])

        # initialize loaders
        train_sampler = SequenceLengthBatchSampler(
            dataset["train"],
            boundaries=boundaries,
            batch_sizes=batch_sizes,
            input_key="input_ids",
            label_key="labels",
        )

        valid_sampler = SequenceLengthBatchSampler(
            dataset["val"],
            boundaries=boundaries,
            batch_sizes=batch_sizes,
            input_key="input_ids",
            label_key="labels",
        )

        test_sampler = SequenceLengthBatchSampler(
            dataset["test"],
            boundaries=boundaries,
            batch_sizes=batch_sizes,
            input_key="input_ids",
            label_key="labels",
        )

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

        # define data loaders
        train_loader = DataLoader(
            dataset["train"],
            batch_size=batch_size,
            collate_fn=partial(pad_collate, padding_value=tokenizer.pad_token_id),
            num_workers=num_workers,
            pin_memory=True if device in ["cuda", "gpu"] else False,
            shuffle=True,
        )
        valid_loader = DataLoader(
            dataset["val"],
            batch_size=batch_size,
            collate_fn=partial(pad_collate, padding_value=tokenizer.pad_token_id),
            num_workers=num_workers,
            pin_memory=True if device in ["cuda", "gpu"] else False,
        )
        test_loader = DataLoader(
            dataset["test"],
            batch_size=batch_size,
            collate_fn=partial(pad_collate, padding_value=tokenizer.pad_token_id),
            num_workers=num_workers,
            pin_memory=True if device in ["cuda", "gpu"] else False,
        )

    return train_loader, valid_loader, test_loader
