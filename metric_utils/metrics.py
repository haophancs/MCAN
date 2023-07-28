import torch
import config
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from data_utils.vqa_vocab import VQAVocab

class Metrics(object):
    def __init__(self, vocab: VQAVocab=None):
        self.vocab = vocab

    def get_scores(self, predicted, true):
        """ Compute the accuracies, precision, recall and F1 score for a batch of predictions and answers """

        predicted = self.vocab._decode_answer(predicted)
        true = self.vocab._decode_answer(true)

        acc = accuracy_score(true, predicted)
        pre = precision_score(true, predicted, average="macro", zero_division=0)
        recall = recall_score(true, predicted, average="macro", zero_division=0)
        f1 = f1_score(true, predicted, average="macro")

        return {
            "accuracy": acc,
            "precision": pre,
            "recall": recall,
            "F1": f1,
            "ground_answers": true,
            "pred_answers": predicted
        }
