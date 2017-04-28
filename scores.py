from sklearn.metrics import accuracy_score, roc_auc_score

def score_function(y_true, y_pred):
    return (roc_auc_score(y_true , y_pred))

def score_accuracy(y_true, y_pred):
    return(accuracy_score(y_true, y_pred))