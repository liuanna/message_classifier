import json
import random
from message_classifier.message_classifier import MessageClassifier


def load_data(input_path):
    with open(input_path) as json_file:
        train_data = json.load(json_file)
    random.shuffle(train_data)
    messages, labels = zip(*train_data)
    return messages, labels


if __name__ == '__main__':
    data, labels = load_data("data/trainingSet.json")
    classifier = MessageClassifier()
    classifier.train(data, labels, use_model=True)
    new_messages = ["how to save more money?", "how to spend less in credit card?"]
    classifier.predict(new_messages)
