from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score

dataset = load_dataset("json", data_files={"data": "ERROS_LOGICOS_DATASET.jsonl"})
full_dataset = dataset["data"].train_test_split(test_size=0.2)

tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = RobertaForSequenceClassification.from_pretrained("microsoft/graphcodebert-base", num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples["code"], truncation=True, padding="max_length", max_length=64)
tokenized_datasets = full_dataset.map(tokenize_function, batched=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


training_args = TrainingArguments(
    output_dir="./results",
    save_steps=100,                 
    save_total_limit=1,            
    learning_rate=5e-5,             
    per_device_train_batch_size=16, 
    num_train_epochs=1,             
    weight_decay=0.01,
    fp16=True                       
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

trainer.train()


model.save_pretrained("./graphcodebert-erro-logico-model")
tokenizer.save_pretrained("./graphcodebert-erro-logico-model")
