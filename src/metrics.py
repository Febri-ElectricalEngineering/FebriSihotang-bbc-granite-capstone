from sklearn.metrics import f1_score, classification_report, confusion_matrix
from rouge_score import rouge_scorer

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")

def clf_report(y_true, y_pred, digits=3):
    return classification_report(y_true, y_pred, digits=digits)

def confmat(y_true, y_pred, labels=None):
    return confusion_matrix(y_true, y_pred, labels=labels)

def rouge_scores(refs, hyps):
    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    R = {"rouge1":[], "rouge2":[], "rougeL":[]}
    for r,h in zip(refs, hyps):
        s = scorer.score(r, h)
        for k in R: R[k].append(s[k].fmeasure)
    return {k: sum(v)/len(v) for k,v in R.items()}
