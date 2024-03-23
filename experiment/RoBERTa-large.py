# Reference from https://huggingface.co/klue/roberta-large
# Reference from https://huggingface.co/bespin-global/klue-sroberta-base-continue-learning-by-mnr
import math
from datetime import datetime
from datasets import load_dataset
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import SentenceTransformer, models, losses

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
val_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    sts_valid,
    name="sts-valid",
)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    sts_test,
    name="sts-test"
)

embedding = models.Transformer(
    model_name_or_path="klue/roberta-large",
    max_seq_length=256,
    do_lower_case=True
)
pooling_model = models.Pooling(
    embedding.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False,
)
model = SentenceTransformer(modules=[embedding, pooling_model])

num_epochs = 4
batch_size = 32
output_path = "save/"+"klue-roberta-large"+"-"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

loss= losses.CosineSimilarityLoss(model=model)
warmup_steps = math.ceil(len(sts_train) * num_epochs / batch_size * 0.1)
model.fit(
    train_objectives=[(train_dataloader, loss)],
    evaluator=val_evaluator,
    epochs=num_epochs,
    evaluation_steps=int(len(train_dataloader)*0.1),
    warmup_steps=warmup_steps,
    output_path=output_path
)

test_evaluator(model, output_path=output_path)
