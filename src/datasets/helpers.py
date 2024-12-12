
from transformers import DataCollatorWithPadding
import evaluate



def get_data_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer=tokenizer)

def get_compute_metrics():
    metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    return compute_metrics