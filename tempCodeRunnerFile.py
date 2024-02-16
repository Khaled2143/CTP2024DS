from transformers import pipeline

classifier = pipeline("sentiment-analysis")

res = classifier("I've veebn waiting for a HuggingFace course my whole life.")

print(res)