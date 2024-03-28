# Reference from https://huggingface.co/klue/roberta-large
# Reference from https://huggingface.co/bespin-global/klue-sroberta-base-continue-learning-by-mnr

import random
import math
import numpy as np
import logging
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, models, LoggingHandler, losses, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from sentence_transformers.datasets import NoDuplicatesDataLoader

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

model_name = 'klue/roberta-large'
nli_num_epochs = 1
sts_num_epochs = 4
batch_size = 32

nli_model_save_path = 'output/training_nli_by_MLRloss_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
sts_model_save_path = 'output/training_sts_continue_training'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

klue_nli_train = load_dataset("klue", "nli", split="train")
print(len(klue_nli_train))
print(klue_nli_train[0])

