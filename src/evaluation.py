import itertools
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt


def load_model(model_path):
    """
    Loads a trained model from the specified path.

    :param model_path: str, path to the saved model file
    :return: trained model
    """
    return joblib.load(model_path)


def plot_confusion_matrix(y_true, y_pred):
    """
    Plots the confusion matrix.

    :param y_true: actual target values
    :param y_pred: predicted target values
    :return: None, displays the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = range(2)
    plt.xticks(tick_marks, range(2))
    plt.yticks(tick_marks, range(2))

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_roc_curve(y_true, y_scores):
    """
    Plots the ROC curve.

    :param y_true: actual target values
    :param y_scores: target scores, probabilities for the positive class
    :return: None, displays the ROC curve
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    model = load_model('path_to_saved_model.joblib')
    X_test, y_test = ...  # Load your test dataset here
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_scores)
