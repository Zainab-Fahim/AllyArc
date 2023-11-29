in order to get web scraped content for AllyArc, i first need to build a model that can extract given the contents of a website the information regarding autism only.

below is one way to code it step-by-step:

1. Import dependencies and load pre-trained model:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "roberta-base"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

2. Load dataset and tokenize texts:

```python 
import datasets
dataset = load_dataset("my_autism_site_dataset") #custom dataset 

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True) 

tokenized_dataset = dataset.map(tokenize, batched=True)
```

3. Prepare data loaders for training:

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(tokenized_dataset["train"], batch_size=16, shuffle=True)
val_loader = DataLoader(tokenized_dataset["test"], batch_size=16)
```

4. Define model training loop with optimizer, loss etc.:

```python
import torch.nn as nn
optimizer = torch.optim.Adam(model.parameters())
loss = nn.BCEWithLogitsLoss()

def train_epoch():
    for batch in train_loader:
        optimizer.zero_grad()
        labels = batch['labels'].type(torch.FloatTensor)
        outputs = model(**batch)
        train_loss = loss(outputs, labels)
        train_loss.backward()
        optimizer.step()
```

5. Train model for N epochs and validate performance:

```python
epochs = 10
for epoch in range(epochs):
    train_epoch()
    val_loss = validate() #run model on val_loader
    print(f"Epoch {epoch}: Train loss {train_loss} | Val loss {val_loss}") 
```


Here is an example of what the autism website dataset could look like:

The dataset should contain:

1. The full text content of each website/webpage document (`text` column)
2. Span annotations that identify parts of the text relevant to autism (`spans` column)

For example:

```
{
  "text": "New research study investigates genetics of autism in South Asian populations. Researchers at University X recruited 350 families with autistic children...",
  
  "spans": [{"start": 0, "end": 114, "label": 1}, 
            {"start": 152, "end": 192, "label": 1}] 
},
{
  "text": "Our learning center provides 10 evidence-based social skills strategies for autistic teenagers to make and keep friends...",
   
  "spans": [{"start": 96, "end": 129, "label": 1}]
}
```

In this example dataset:

- Each row contains one website document
- The "text" column has the full raw text content
- "spans" column lists labeled text segments relevant to autism
- Spans have "start", "end" and "label" (1 = positive, 0 = negative)

Ideally the dataset should have 500+ examples with the text and span annotations to train the model effectively.


where, the `start` and `end` values in the span annotations refer to character indexes indicating the position of the span in the overall website text.

For example:

```
{
  "text": "New autism research study investigates genetics...",
  
  "spans": [{"start": 4, "end": 19, "label": 1}]
}
```

Here:

- `text` length is 44 characters
- The annotated span goes from index 4 to index 19 in the text
- So it is marking the span "autism research" as being relevant

The start and end allow marking which specific part of the long text is considered important for the task, rather than just labeling the full text.

Typically, datasets will have multiple spans marked in each text to capture multiple relevant keywords/phrases.

The model will then learn to predict these spans when given new texts during inference.


