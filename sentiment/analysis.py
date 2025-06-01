from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay


def analyze_results(results, model):
    '''
    Analyzes the result of the model and prints the classification report & confusion matrix.
    
    :input: results: the results of the model, which is a tuple of (y_pred, y_true) or (y_pred, y_true, sentences)
    :input: model: the name of the model, e.g. 'Logistic Regression' or 'VADER'
    '''

    y_pred, y_true = results

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
    '''
    Compares the predictions of the Logistic Regression and VADER models, printing out each sentence and
    the predicted labels for both models as well as the true labels. Also plots the accuracy of the two models.

    :input: logreg_preds: predictions from the Logistic Regression model
    :input: vader_preds: predictions from the VADER model
    :input: labels: true labels for the sentences
    :input: sentences: the sentences data
    '''

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
    
    plt.figure(figsize=(16, 8))
    plt.subplot(1,2,1) 
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