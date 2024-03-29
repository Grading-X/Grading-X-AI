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

def triplet(dataset):

  train_data = {}
  def dataset_construction(hypothesis, premise, label):
    if hypothesis not in train_data:
      train_data[hypothesis] = {'contradiction':set(), 'entailment': set(), 'neutral': set()}
    train_data[hypothesis][label].add(premise)

  for i, data in enumerate(dataset):
    hypothesis = data['hypothesis'].strip()
    premise = data['premise'].strip()
    if data['label'] == 0:
      label = 'entailment'
    elif data['label'] == 1:
      label = 'neutral'
    else:
      label = 'contradiction'

    dataset_construction(hypothesis, premise, label)
    dataset_construction(premise, hypothesis, label)

  input_examples = []
  for hypothesis, others in train_data.items():
    if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
      input_examples.append( InputExample(texts=[hypothesis, random.choice(list(others['entailment'])), random.choice(list(others['contradiction']))]) )
      input_examples.append( InputExample(texts=[random.choice(list(others['entailment'])), hypothesis, random.choice(list(others['contradiction']))]) )

  return input_examples

nli_train_examples = triplet(klue_nli_train)
print(nli_train_examples[0].texts)
print(nli_train_examples[1].texts)

nli_train_dataloader = NoDuplicatesDataLoader(
    nli_train_examples,
    batch_size=batch_size,
)