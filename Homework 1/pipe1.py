from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch 
import torch.nn.functional as F

# Applying the Pipeline
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model,tokenizer=tokenizer)

# Using multiple sentences, our data and feed it to the pipeline classifier then print the data
X_train = ["I've been waiting for some cheetos my whole life.", "Python is super cool"]

res = classifier(X_train)
print(res)

# Calling tokenizer with our data called batch
batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt") # Pytorch format
print(batch)

# Do the infrence in pytorch
with torch.no_grad():
    outputs = model(**batch)  
    print(outputs)
    predictions = F.softmax(outputs.logits, dim=1)
    print(predictions)
    labels = torch.argmax(predictions, dim=1)
    print(labels)