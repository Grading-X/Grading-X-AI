# Reference from https://huggingface.co/bespin-global/klue-sroberta-base-continue-learning-by-mnr

from datasets import load_dataset
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

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

batch_size = 16
train_dataloader = DataLoader(
    sts_train,
    shuffle=True,
    batch_size=batch_size,
)

evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    sts_valid,
    name="sts-valid",
)

test_evaluator =  EmbeddingSimilarityEvaluator.from_input_examples(
    sts_test,
    name="sts-test"
)