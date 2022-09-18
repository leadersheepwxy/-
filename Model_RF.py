from sklearn.model_selection import StratifiedKFold
import pandas as pd
import statistics
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import plot_roc_curve, roc_curve
from sklearn import tree
import settingFunction as sf

def RF_model(block, normal, ax_val, ax_test):
    criterion_type = 'gini'  #
    estimators_value_range = range(1, 101)

    X, y, X_training, X_testing, y_training, y_testing = sf.DataImport(normal, block)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)  # 定義10折交叉驗證:分層KFold 90/10
    i, j = 1, 1
    bestMean = [0, 0, 0, 0]
    t_bestMean = [0, 0, 0, 0]
    bestGroup, bestParm = 0, 0
    bestAuc, bestRecall, bestPrec, bestF1 = 0, 0, 0, 0
    t, v = [], []
    meanAuc_list = []

    for n in estimators_value_range:
        auc_list, recall_list, prec_list, f1_list = [], [], [], []
        t_auc_list, t_recall_list, t_prec_list, t_f1_list = [], [], [], []
        train_index_list, val_index_list = [], []
        fpr_list, tpr_list = [], []

        for train_index, val_index in cv.split(X_training, y_training):
            X_train, X_val = X_training[train_index], X_training[val_index]
            y_train, y_val = y_training[train_index], y_training[val_index]

            model = RandomForestClassifier(random_state=i, n_estimators=n, max_depth=3)
            model.fit(X_train, y_train)  # 訓練
            predictions_train = model.predict(X_train)
            predictions_val = model.predict(X_val)
            # print("-------第", i, "組-------")
            # sf.ScoreReport('y', y_val, predictions_val)

            fpr, tpr, thresholds = roc_curve(y_val, predictions_val)
            fpr_list.append(fpr)
            tpr_list.append(tpr)

            t_accuracy = sklearn.metrics.accuracy_score(y_train, predictions_train)
            t_recall = sklearn.metrics.recall_score(y_train, predictions_train)
            t_precision = sklearn.metrics.precision_score(y_train, predictions_train)
            t_f1 = sklearn.metrics.f1_score(y_train, predictions_train)
            t_auc_list.append(t_accuracy)
            t_recall_list.append(t_recall)
            t_prec_list.append(t_precision)
            t_f1_list.append(t_f1)

            accuracy = sklearn.metrics.accuracy_score(y_val, predictions_val)
            recall = sklearn.metrics.recall_score(y_val, predictions_val)
            precision = sklearn.metrics.precision_score(y_val, predictions_val)
            f1 = sklearn.metrics.f1_score(y_val, predictions_val)
            auc_list.append(accuracy)
            recall_list.append(recall)
            prec_list.append(precision)
            f1_list.append(f1)
            train_index_list.append(train_index)
            val_index_list.append(val_index)

            if (j == 10):
                t_meanAuc = statistics.mean(t_auc_list)
                meanAuc = statistics.mean(auc_list)
                meanAuc_list.append(meanAuc)

                # print("第", int(i / 10), "組 結果:", auc_list)
                if meanAuc > bestMean[0]:
                    # train
                    t_meanRecall = statistics.mean(t_recall_list)
                    t_meanPrec = statistics.mean(t_prec_list)
                    t_meanF1 = statistics.mean(t_f1_list)
                    t_bestMean = (t_meanAuc, t_meanRecall, t_meanPrec, t_meanF1)

                    # val
                    meanRecall = statistics.mean(recall_list)
                    meanPrec = statistics.mean(prec_list)
                    meanF1 = statistics.mean(f1_list)
                    bestMean = (meanAuc, meanRecall, meanPrec, meanF1)

                    bestAuc_index = auc_list.index(max(auc_list))
                    bestAuc = auc_list[bestAuc_index]

                    t = train_index_list[bestAuc_index]
                    v = val_index_list[bestAuc_index]

                    bestGroup = ((bestAuc_index + 1) + (i - 10))
                    bestParm = n
                j = 0
            j += 1
            i += 1
    foldNum = (bestGroup % 10)
    if foldNum == 0:
        foldNum == 10

    print("=========================Data切割狀況=========================")
    print("--TRAINING SET--")
    print(X_training.shape)
    print("-train-")
    print(X_train.shape)
    print("-validation-")
    print(X_val.shape)
    print("--TESTING SET--")
    print(X_testing.shape)

    print("=========================參數最佳化=========================")
    bestValue = [bestGroup, bestAuc, bestParm]
    bestValue_name = ['最佳組別:', '最佳準確率:', '最佳C:']  #
    bestValue_Chart = pd.DataFrame(bestValue, bestValue_name)
    print(bestValue_Chart)

    print("=========================預測結果=========================")
    print("* 平均最佳： NO.", bestGroup, " (第", int(bestGroup / 10 + 1), "組, fold", foldNum, ") *")
    model = RandomForestClassifier(random_state=bestGroup, n_estimators=bestParm, max_depth=3)
    X_train, X_val = X_training[t], X_training[v]
    y_train, y_val = y_training[t], y_training[v]
    model.fit(X_train, y_train)  # 訓練

    print("\n-TRAIN-")
    predictions_train = model.predict(X_train)
    train_score = sf.ScoreReport(y_train, predictions_train)

    print("\n-Validation-")
    predictions_val = model.predict(X_val)
    val_score = sf.ScoreReport(y_val, predictions_val)

    print("\n-TEST-")
    predictions_test = model.predict(X_testing)
    test_score = sf.ScoreReport(y_testing, predictions_test)

    Scores = {
        "Score": ['accuracy', 'recall', 'precision', 'f1'],
        "Train(mean)": t_bestMean,
        "Val(mean)": bestMean,
        "Train(best)": train_score,
        "Val(best)": val_score,
        "Test": test_score
    }
    ScoresChart = pd.DataFrame(Scores)
    print(ScoresChart)

    RF_roc_val = plot_roc_curve(model, X_val, y_val, ax=ax_val, color='green')
    # plt.show()
    RF_roc_test = plot_roc_curve(model, X_testing, y_testing, ax=ax_test, color='green')
    # plt.show()

    '''
    plt.plot(list(estimators_value_range), meanAuc_list)
    plt.xlabel('n estimators')
    plt.ylabel('Accuracy')  # 通过图像选择最好的参数
    # plt.xticks(list(estimators_value_range), list(estimators_value_range))
    plt.show()

    fn = X.columns
    cn = ['0', '1']
    tree.plot_tree(model.estimators_[bestParm - 1], feature_names=fn, class_names=cn, filled=True, fontsize=6.5)
    plt.show()

    print("=========================特徵重要性=========================")
    import_level_list = []
    x_columns_list = []
    import_level = model.feature_importances_  # 這個方法可以調取關於特徵重要程度
    x_columns = X.columns[0:]
    index = np.argsort(import_level)[::-1]
    for each in range(X.shape[1]):
        import_level_value = import_level[index[each]]
        x_columns_value = x_columns[each]
        print('The important level of ' + x_columns_value + ':      ' + str(import_level_value))
        import_level_list.append(import_level_value)
        x_columns_list.append(x_columns_value)

    sorted_idx = model.feature_importances_.argsort()
    plt.barh(np.array(x_columns_list)[sorted_idx], np.array(model.feature_importances_)[sorted_idx])
    plt.xlabel("Random Forest Feature Importance")
    plt.show()
    '''
if __name__ == '__main__':
    block = pd.read_csv(
        "C:\\Users\\wxy\\PycharmProjects\\project108\\AVFflow\\0_dataSet.csv")
    normal = pd.read_csv(
        "C:\\Users\\wxy\\PycharmProjects\\project108\\AVFflow\\1_dataSet.csv")
    fig, axes = plt.subplots(1, 2)
    ax_val = axes[0]
    ax_test = axes[1]

    RF_model(block, normal, ax_val, ax_test)
    plt.show()