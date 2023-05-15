import itertools
import numpy as np
from torchmetrics.functional import auroc, roc
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from matplotlib import pyplot as plt


class Metrics:

    @staticmethod
    def get_confusion_matrix(y_true, y_pred):
        return confusion_matrix(y_true.cpu(), y_pred.cpu())

    @staticmethod
    def get_classification_report_text(y_true, y_pred, classes):
        print("Classification report: (NOTE THAT FOR MULTI-CLASS ONE-LABEL CLASSIFICATION:"
              " MICRO-AVG ACCURACY == MICRO-AVG PRECISION == MICRO-AVG RECALL == MICRO-AVG F1-SCORE")
        return classification_report(y_true.cpu(), y_pred.cpu(), target_names=classes)

    @staticmethod
    def get_classification_report_data(y_true, y_pred, classes):
        data = {}
        data["micro_precision"], data["micro_recall"], data["micro_fscore"], _ = precision_recall_fscore_support(y_true.cpu(), y_pred.cpu(), average="micro")
        data["macro_precision"], data["macro_recall"], data["macro_fscore"], _ = precision_recall_fscore_support(y_true.cpu(), y_pred.cpu(), average="macro")
        data["weighted_precision"], data["weighted_recall"], data["weighted_fscore"], _ = precision_recall_fscore_support(y_true.cpu(), y_pred.cpu(), average="weighted")

        precision, recall, fscore, support = precision_recall_fscore_support(y_true.cpu(), y_pred.cpu())
        for i, label in enumerate(classes):
            data[label] = {}
            data[label]["precision"] = precision[i]
            data[label]["recall"] = recall[i]
            data[label]["fscore"] = fscore[i]
            data[label]["support"] = support[i]
        return data

    @staticmethod
    def get_micro_accuracy(cm):
        return np.trace(cm) / np.sum(cm)

    @staticmethod
    def get_macro_accuracy(cm):
        digonal_sum = np.trace(cm)
        diagonal = np.diag(cm)
        rows_sum = np.sum(cm, axis=1)
        cols_sum = np.sum(cm, axis=0)
        accuracies = digonal_sum / (digonal_sum + rows_sum + cols_sum - 2 * diagonal)
        return np.mean(accuracies)

    @staticmethod
    def get_auc_per_class(y_true, y_probas, num_classes):
        return auroc(y_probas, y_true.long(), "multiclass", num_classes=num_classes, average=None)

    @staticmethod
    def get_fpr_tpr(y_true, y_probas, num_classes):
        fpr, tpr, _ = roc(y_probas, y_true.long(), "multiclass", num_classes=num_classes)
        return fpr, tpr

    @staticmethod
    def plot_pretty_confusion_matrix(y_true, y_pred, classes):
        cm = confusion_matrix(y_true.cpu(), y_pred.cpu())
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.canvas.draw()
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax)

        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)

        ax.set(title='Confusion matrix', xlabel='Predicted label', ylabel='True label')

        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        ax.yaxis.label.set_size(20)
        ax.xaxis.label.set_size(20)
        ax.title.set_size(25)

        threshold = (cm.max() + cm.min()) / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                     horizontalalignment='center',
                     color='white' if cm[i, j] > threshold else 'black',
                     size=10)
        plt.show()

    @staticmethod
    def plot_roc_curves_torch(fpr, tpr, auc_per_class, classes_names):
        colors = ['blue', 'red', 'green', 'orange', 'black']
        plt.figure(figsize=(12, 7))
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        for i, color in enumerate(colors):
            plt.plot(fpr[i].cpu(), tpr[i].cpu(), color=color, lw=2,
                     label=f'ROC curve of class {classes_names[i]} (AUC = {auc_per_class[i]:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve for multiclass classification')
        plt.legend(loc="lower right")
        plt.show()
