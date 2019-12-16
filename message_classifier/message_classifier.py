import spacy
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from spacy.util import minibatch, compounding


class MessageClassifier:
    def __init__(self, n_iter:int = 20, test_size: float = 0.2, output_dir: str = "./model"):
        """
        Initiate a MessageClassifier object
        :param n_iter: int; defalut 20
            number of iterations for training
        :param test_size: float; default 0.2
            proportion of test set (0-1)
        :param output_dir: str
            a path to save or retrieve trained model
        """
        self.n_iter = n_iter
        self.model_path = output_dir
        self.test_size = test_size
        self._check_output_dir()

    def _check_output_dir(self):
        """create output folder if it does not exist"""
        if self.model_path is None or not os.path.isdir(self.model_path):
            self.model_path = "./model"
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

    @staticmethod
    def _validate(data: list):
        """Convert all entries in the list to string"""
        return [str(entry).lower() for entry in data]

    def _prepare_data(self, classifier, data, labels):
        """split data into train/test dataset for model"""
        converted_labels = [{cat: False for cat in classifier.labels} for _ in labels]
        for i, label in enumerate(labels):
            converted_labels[i][label] = True
        # split train and test data
        xTrain, xTest, yTrain, yTest = train_test_split(data, converted_labels, test_size=self.test_size, random_state=0)
        return xTrain, xTest, yTrain, yTest

    def _load_model(self, use_model: bool):
        """load pre-trained model if exists; otherwise create a new model"""
        if os.path.exists(os.path.join(self.model_path, 'textcat')) and use_model:
            nlp = spacy.load(self.model_path)
        else:
            nlp = spacy.blank("en")

        # add the text classifier to the pipeline if it doesn't exist
        if "textcat" not in nlp.pipe_names:
            textcat = nlp.create_pipe(
                "textcat", config={"exclusive_classes": True, "architecture": "simple_cnn"}
            )
            nlp.add_pipe(textcat, last=True)
        # otherwise, get it, so we can add labels to it
        else:
            textcat = nlp.get_pipe("textcat")
        return nlp, textcat

    def _train(self, data: list, labels: list, use_model: bool = True, save_model: bool = True):
        """
        Private method for training a message classifier
        :param data: list
            a list of strings (ex. ['message1', 'message2', ...])
        :param labels: list
            a list of labels corresponding to each entry of data, must be same length as data (ex. ['label1', 'label2', ...])
        :param use_model: bool; default True
            Use existing model if available
        :param save_model: bool; default True
            If True, the trained model will be saved to disk
        """
        assert len(data) == len(labels), f"data ({len(data)}) and labels ({len(labels)}) have inconsistent length"

        # validate data and labels to be a list of strings
        data = self._validate(data)
        labels = self._validate(labels)

        nlp, textcat = self._load_model(use_model)
        # add new label to text classifier
        for label in set(labels):
            if label not in textcat.labels:
                textcat.add_label(label)
        # Prepare training and test data
        train_texts, test_texts, train_cats, test_cats = self._prepare_data(textcat, data, labels)
        train_data = list(zip(train_texts, [{"cats": cats} for cats in train_cats]))

        optimizer = nlp.begin_training()
        print("Training the model...")
        print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("Loss", "Precision", "Recall", "F1 Score"))
        batch_sizes = compounding(4.0, 32.0, 1.001)
        for i in range(self.n_iter):
            losses = {}
            # batch up the examples using spaCy's minibatch
            random.shuffle(train_data)
            batches = minibatch(train_data, size=batch_sizes)
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = self._evaluate(nlp.tokenizer, textcat, test_texts, test_cats)
            print(
                "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(  # print a simple table
                    losses["textcat"],
                    scores["textcat_p"],
                    scores["textcat_r"],
                    scores["textcat_f"],
                )
            )
        if save_model:
            self._save(nlp, optimizer.averages)

    def train(self, data: list, labels: list, use_model: bool = True, save_model: bool = True):
        """
        Train a message classifier
        :param data: list
            a list of strings (ex. ['message1', 'message2', ...])
        :param labels: list
            a list of labels corresponding to each entry of data, must be same length as data (ex. ['label1', 'label2', ...])
        :param use_model: bool; default True
            Use existing model if available
        :param save_model: bool; default True
            If True, the trained model will be saved to disk
        """
        self._train(data, labels, use_model, save_model)

    def _save(self, model, parameters):
        """save model to disk"""
        if self.model_path is not None:
            with model.use_params(parameters):
                model.to_disk(self.model_path)

    @staticmethod
    def _evaluate(tokenizer, textcat, texts, cats):
        """Evaluate models"""
        true_labels = [max(cat, key=cat.get) for cat in cats]
        docs = (tokenizer(text) for text in texts)
        predictions = [max(doc.cats, key=doc.cats.get) for doc in textcat.pipe(docs)]
        precision = precision_score(true_labels, predictions, average="weighted")
        recall = recall_score(true_labels, predictions, average="weighted")
        f_score = f1_score(true_labels, predictions, average="weighted")
        return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}

    def _predict(self, new_messages: list):
        """
        Private method for predicting on new messages
        :param new_messages: list
            a list of new messages for predictions
        :return: list
            a list of predicted categories corresponding each input new message
        """
        nlp = spacy.load(self.model_path)
        predictions = [max(doc.cats, key=doc.cats.get) for doc in nlp.pipe(new_messages)]
        return predictions

    def predict(self, new_messages: list):
        """
        Predict on new messages
        :param new_messages: list
            a list of new messages for predictions
        :return: list
            a list of predicted categories corresponding each input new message
        """
        return self._predict(new_messages)
