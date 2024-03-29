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

klue_sts_train = load_dataset("klue", "sts", split='train[:90%]')
klue_sts_valid = load_dataset("klue", "sts", split='train[-10%:]')
klue_sts_test = load_dataset("klue", "sts", split="validation")
print(len(klue_sts_train))
print(len(klue_sts_valid))
print(len(klue_sts_test))
print(klue_sts_train[0])

def preprocessing(dataset):
    result = []
    for i, data in enumerate(dataset):
        sentence1 = data['sentence1']
        sentence2 = data['sentence2']
        score = (data['labels']['label'] / 5.0)  # 5점 만점, 0~5점으로 정규화

        result.append(InputExample(texts=[sentence1, sentence2], label=score))
    return result

sts_train = preprocessing(klue_sts_train)
sts_valid = preprocessing(klue_sts_valid)
sts_test = preprocessing(klue_sts_test)

train_dataloader = DataLoader(
    sts_train,
    shuffle=True,
    batch_size=batch_size,
)
val_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    sts_valid,
    name="sts-valid",
)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    sts_test,
    name="sts-test"
)

embedding = models.Transformer(
    model_name_or_path = 'klue/roberta-large',
    max_seq_length=256,
    do_lower_case=True
)

pooling = models.Pooling(
    embedding.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False,
)

model = SentenceTransformer(modules=[embedding, pooling])

train_loss = losses.MultipleNegativesRankingLoss(model)

warmup_steps = math.ceil(len(nli_train_examples)*nli_num_epochs/batch_size*0.1)
logging.info("Warmup-steps: {}".format(warmup_steps))

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=val_evaluator,
    epochs=nli_num_epochs,
    evaluation_steps=int(len(nli_train_dataloader)*0.1),
    warmup_steps=warmup_steps,
    output_path=nli_model_save_path,
    use_amp=False,
)

model = SentenceTransformer(nli_model_save_path)

train_loss = losses.CosineSimilarityLoss(model=model)

warmup_steps = math.ceil(len(sts_train)*sts_num_epochs/batch_size*0.1)
logging.info("Warmup-steps: {}".format(warmup_steps))

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=val_evaluator,
    epochs=sts_num_epochs,
    evaluation_steps=int(len(train_dataloader)*0.1),
    warmup_steps=warmup_steps,
    output_path=sts_model_save_path
)

print(test_evaluator(model, output_path=sts_model_save_path))