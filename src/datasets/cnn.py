from datasets import load_dataset


    
def CNN(tokenizer):
    
    def tokenization(example):
        inputs = tokenizer(example["article"], padding="max_length", truncation="only_first", max_length=512)
        outputs = tokenizer(example["highlights"], padding="max_length", truncation="only_first", max_length=128)    
        inputs["labels"] = outputs["input_ids"]
        return inputs
        
    dataset = load_dataset("cnn_dailymail","3.0.0")
    dataset_train = dataset['train'].select(range(1000)).map(tokenization, batched=True)
    dataset_val = dataset['validation'].select(range(1000)).map(tokenization, batched=True)
    return {"train":dataset_train, "val":dataset_val}
    