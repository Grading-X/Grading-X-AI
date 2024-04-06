import numpy as np
from transformers import AutoTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, load_metric, Dataset

dataset = load_dataset("csv", data_files="train.csv")['train']
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
max_len = 256

def preprocessing(data):
    result = tokenizer(data['sentence1'], data['sentence2'], padding=True, max_length=max_len, truncation=True)
    result['label'] = data['similar']
    return result

dataset = dataset.map(preprocessing, remove_columns=['sentence1', 'sentence2', 'similar'])
dataset = dataset.train_test_split(0.1)

collator = DataCollatorWithPadding(tokenizer=tokenizer)
metric = load_metric('glue', 'sst2')

def cp_metric(p):
    preds, labels = p
    result = metric.compute(references=labels, predictions=np.argmax(preds, axis=-1))

    return result

model = RobertaForSequenceClassification.from_pretrained("klue/roberta-base")

# 메모리 상황에 맞게 batch 사이즈 조정해야함
batch_size = 8
args = TrainingArguments(
    'runs/',
    per_device_train_batch_size=batch_size,
    num_train_epochs=3,
    do_train=True,
    do_eval=True,
    save_strategy="epoch",
    logging_strategy="epoch",
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=cp_metric
)

trainer.train()

# 전체 Inference
test = load_dataset("csv", data_files="test.csv")['train']
test = test.map(preprocessing, remove_columns=['sentence1', 'sentence2'])
predictions = trainer.predict(test)

# 일부만 Inference
num_data = 30
data = Dataset.from_dict(test[:num_data])
predictions = trainer.predict(data)
result = np.argmax(predictions.predictions, axis=-1)
