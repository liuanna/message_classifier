# Message Classifier

## Intro
**Goal:** To use Machine Learning models to classify messages into specific categories (shown below)

Available categories:
```markdown
ACCOUNT_LINKING
APP_GENERAL
BILLS
BUDGET
CANCELLATION
CASH_TRANSFER
CHARGES
DEBT
FINANCIAL_PLANNING
FINANCIAL_TIPS
INSURANCE
INVESTING
NOT_SUPPORTED_ADVICE
SAVING
SPENDING
```

## Installation
1) Clone this repository to your local and navigate into the package where the `setup.py` locates in the command line tool
2) Run the following line to install the package
```shell script
python setup.py install
```

## Quick Start
**Loading packages**
```python
import json
import random
from message_classifier.message_classifier import MessageClassifier
```
The `MessageClassifier` accepts a list of text messages as data and a list of categories as labels

Below is a sample utility function to load data from a json file in the format `[[message1, label1],[message2, label2], ...]`
```python
def load_data(input_path):
    with open(input_path) as json_file:
        train_data = json.load(json_file)
    random.shuffle(train_data)
    messages, labels = zip(*train_data)
    return messages, labels
```
**Train the model**

Parameters:
* `data`: a list of message strings `['message1', 'message2', ...]`
* `label`: a list of category, corresponding to each message `['label1', 'label2', ...]`
* `use_model`: True/False; if True, will train the model on the top of the pre-trained model
* `save_model`: True/False; if True, will save the model

```python
data, labels = load_data("data/trainingSet.json")
classifier = MessageClassifier()
classifier.train(data, labels, use_model=True)
```
**Test on new messages**
```python
new_messages = ["how to save more money?", "how to spend less in credit card?"]
classifier.predict(new_messages)
```


