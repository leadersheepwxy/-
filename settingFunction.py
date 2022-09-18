import pandas as pd
import numpy as np
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

auc_val_value, recall_val_value, prec_val_value, f1_val_value, cm_val_value = [], [], [], [], []


def DataImport(normal_csv, block_csv):
    block = block_csv
    normal = normal_csv
    data = normal.append(block)

    X = data.drop('target', axis=1)  # 刪除target列之欄位
    y = data['target']
    '''
    #剔除射頻功率
    X = X.drop('p1', axis=1)
    X = X.drop('p2', axis=1)
    X = X.drop('p3', axis=1)
    X = X.drop('p4', axis=1)
    '''
    variables = X.values
    type_label = (data['target']).values
    X_training, X_testing, y_training, y_testing = train_test_split(variables, type_label, test_size=0.2,
                                                                    random_state=1)  # 80% training and 20% test
    return X, y, X_training, X_testing, y_training, y_testing


def ScoreChart(mode, accuracy, recall, precision, f1):
    mode = str(mode)
    t_scores = [accuracy, recall, precision, f1]
    t_scores_name = ['accuracy', 'recall', 'precision', 'f1']
    t_scores_chart = pd.DataFrame(t_scores, t_scores_name)
    if 'y' in mode:
        print(t_scores_chart)



def ScoreReport(y_t, predictions_t):
    accuracy = sklearn.metrics.accuracy_score(y_t, predictions_t)
    recall = sklearn.metrics.recall_score(y_t, predictions_t)
    precision = sklearn.metrics.precision_score(y_t, predictions_t)
    f1 = sklearn.metrics.f1_score(y_t, predictions_t)
    cm = confusion_matrix(y_t, predictions_t)
    print(cm)
    t_scores = [accuracy, recall, precision, f1]
    return t_scores


def ScoreList(mode, y_t, predictions_t):
    mode = str(mode)
    accuracy, recall, precision, f1, cm = ScoreReport(mode, y_t, predictions_t)
    auc_val_value.append(accuracy)
    recall_val_value.append(recall)
    prec_val_value.append(precision)
    f1_val_value.append(f1)
    cm_val_value.append(cm)
    return auc_val_value, recall_val_value, prec_val_value, f1_val_value, cm_val_value


def ParmFig(parmName,start,end,auc_val):
    parmName = str(parmName)
    axis_parm = list(range(start, end+1))
    axis_auc = []
    for i in range(0, end):
        j = (i * 10)
        axis_auc.append(max(auc_val[(0 + j):(10 + j)]))
    plt.plot(axis_parm, axis_auc, marker="o")
    plt.xlabel(parmName)
    plt.ylabel('Accuracy')  # 通过图像选择最好的参数
    plt.xticks(axis_parm, axis_parm)
    plt.show()

def roc(title,model, X_testing, y_testing):
    # fpr, tpr, thresholds = roc_curve(y_t, predictions_t)
    # roc_auc = auc(fpr, tpr)
    # plt.plot(fpr, tpr, label='%s (AUC = %0.2f)' % (title, roc_auc))
    y_score = model.decision_function(X_testing)
    fpr, tpr, threshold = roc_curve(y_testing, y_score)  ###計算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###計算auc的值
    plt.plot(fpr, tpr, label='%s (AUC = %0.2f)' % (title, roc_auc))  ###假正率為橫座標，真正率為縱座標做曲線

def rocshow(title):
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

tprs,aucs = [], []
mean_fpr = np.linspace(0, 1, 100)
def cvRoc(foldNum, fprList,tprList):
    tprs, aucs = [], []
    for i in range(0, 10):
        tprs.append(np.interp(mean_fpr, fprList[i], tprList[i]))
        tprs[-1][0] = 0.0
        roc_auc = auc(fprList[i], tprList[i])
        aucs.append(roc_auc)
        plt.plot(fprList[i], tprList[i], lw=1, alpha=0.6,
                 label='ROC fold %d (AUC = %0.2f)' % ((i + 1), roc_auc))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2.5,
             alpha=.9)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.1, label=r'$\pm$ 1 std. dev.')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('CrossValidation Roc')
    plt.legend(loc="lower right")
    plt.show()