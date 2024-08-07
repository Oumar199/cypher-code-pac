# -*- coding: utf-8 -*-
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer
)
from transformers import (
    GenerationConfig,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup,
    T5ForConditionalGeneration, Adafactor, BartForConditionalGeneration
)
from wolof_translate.utils.bucket_iterator import SequenceLengthBatchSampler, BucketSampler
from wolof_translate.utils.sent_transformers import TransformerSequences
from wolof_translate.utils.sent_corrections import *
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from nlpaug.augmenter import char as nac
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from functools import partial
import sentencepiece as spm
from math import ceil
import pandas as pd
import numpy as np
import argparse
import evaluate
import string
import random
import shutil
import wandb
import torch
import time
import nltk
import os

# Désactiver le parallélisme des tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
