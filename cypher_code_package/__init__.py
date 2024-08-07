# -*- coding: utf-8 -*-

from transformers import (
    CodeGenForCausalLM,
    CodeGenTokenizerFast,
    GenerationConfig,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup,
)
from wolof_translate.utils.bucket_iterator import SequenceLengthBatchSampler
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from datasets.dataset_dict import DatasetDict
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from functools import partial
from datasets import Dataset
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
