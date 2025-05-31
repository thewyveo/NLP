from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay


def analyze_results(results, model):
    """
    Compare predictions from two sentiment models.
    
    :param results: Tuple containing (logreg_preds, gold_labels), (vader_preds, gold_labels)
    """

    (y_pred, y_true) = results if model != 'VADER' else results

    print(f"\n=== ST - {model} Results ===")
    print(classification_report(y_true, y_pred))
    if model == 'VADER':
        print(f"\t=== ST - Confusion Matrices ===")
    cm = confusion_matrix(y_true, y_pred, labels=['positive', 'negative', 'neutral'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['positive', 'negative', 'neutral'])
    ax = disp.plot(cmap=plt.cm.Blues)
    ax.ax_.set_title(f'{model} Model')
    plt.show()

def compare_models(logreg_preds, vader_preds, labels, sentences):
    print("\n=== ST - Sentences ===")
    for i, (lr_pred, vader_pred, true_label) in enumerate(zip(logreg_preds, vader_preds, labels)):
        if lr_pred != vader_pred:
            if i != 0:
                print()
            print(f"- Sentence: {sentences[i]}")
            print(f"  True: {true_label} | LogReg: {lr_pred} | VADER: {vader_pred}")
            print()    

    vader_accuracy = float(accuracy_score(labels, vader_preds))
    logreg_accuracy = float(accuracy_score(labels, logreg_preds))
    
    # plotting
    plt.figure(figsize=(16, 8))    # set the figure size
    plt.subplot(1,2,1)    # 1 row, 2 columns, first subplot.
    bars = plt.bar(['VADER', 'Logistic Regression'], [vader_accuracy, logreg_accuracy], color=['red', 'blue'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')

    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend(['VADER', 'Logistic Regression'], loc='upper left')
    plt.show()