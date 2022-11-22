
import numpy as np
from sklearn.metrics import *


class BaseScore:

    def __init__(self,**args):
        self.args = args


class AccuracyScore(BaseScore):

    def compute(self,reference,logits):
        predictions = np.argmax(logits,axis=-1)
        return accuracy_score(reference,predictions,**self.args)

        
class FScore(BaseScore):

    def compute(self,reference,logits):
        predictions = np.argmax(logits,axis=-1)
        return fbeta_score(reference,predictions,**self.args)
        

METRICS = {
    "accuracy": AccuracyScore,
    "fscore": FScore,
}

# def sigmoid(x):
#     p = np.where(
#         x >= 0, 
#         1 / (1 + np.exp(-x)),
#         np.exp(x) / (1 + np.exp(x))
#     )
#     return p


# class RiskScore:

#     def __init__(self,L_matrix):
#         self.L_matrix = L_matrix

#     def compute(self,reference,logits):
#         if len(logits.shape) > 1:
#             probs = np.exp(logits - np.max(logits,axis=1,keepdims=True))
#             probs = probs / np.sum(probs,axis=1,keepdims=True)
#         elif len(logits.shape) == 1:
#             probs = sigmoid(logits)
#             probs = np.hstack((probs,1-probs))
#         else:
#             raise ValueError("logits can't be an empty vector")

#         conditional_risk = probs @ self.L_matrix
#         bayes_decisions = np.argmin(conditional_risk,axis=1)
        
        