import matplotlib as plt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plotting(accuracies, data, text):
    '''
    Plotting function to visualize the accuracies of the model across folds, the mean val. accuracy, and the test accuracy.
    :input: accuracies: A tuple containing the mean training accuracy and a history list of the accuracies across folds.
    :input: data: A tuple containing the predicted labels and true labels.
    :input: text: A list of sentences corresponding to the predicted labels.
    :return: test_accuracy: The accuracy of the model on the test set.
    '''

    # extract all params
    training_mean_acc, accs_across_folds = accuracies
    y_pred, y_true = data
    sentences = text

    # printing prediction results
    print('IDX - C/IC - PRED LABEL - TRUE LABEL - TEXT')
    wrongs = 0
    all_y_pred, all_y_true = [], []
    for idx, (y_pred, y_true) in enumerate(zip(y_pred, y_true)):
        if y_pred != y_true:
            print(f'{idx+1} - INcorrect - {y_pred} - {y_true} - {sentences[idx]}')
            wrongs += 1
        else:
            print(f'{idx+1} - correct - {y_pred} - {y_true} - {sentences[idx]}')
        all_y_pred.append(y_pred)
        all_y_true.append(y_true)
    print(f'\n\tTotal incorrect predictions: {wrongs} out of 18')

    labels = ['sports', 'movie', 'book']
    print(f"\n\tConfusion Matrix ('sports', 'movie', 'book'):")
    cm = confusion_matrix(all_y_true, all_y_pred, labels=labels)
    print(cm)

    # calculating test accuracy & setting limits for y-axis
    test_accuracy = ((18-wrongs) / 18)
    limit1, limit2 = 0.35, 0.95    # default limits for y-axis
    if test_accuracy > 0.75:
        limit1 = 0.7
        limit2 = 0.95
    else:
        limit1 = 0.5
        limit2 = 0.95

    # plotting
    plt.figure(figsize=(16, 8))    # set the figure size
    plt.subplot(1,2,1)    # 1 row, 2 columns, first subplot.
    plt.plot(accs_across_folds[0:4], label='Training Accuracy', color='green')
    plt.axhline(y=training_mean_acc, color='red', linestyle='--', label='Mean Validation Accuracy')
    plt.axhline(y=test_accuracy, color='blue', linestyle=':', label='Test Accuracy')
    plt.text(len(accs_across_folds[0:4]), test_accuracy, f'Test: {test_accuracy:.2%}',   # text for parameters of 1 value (not list)
        ha='right', va='bottom', color='blue', fontsize=10)   
    plt.text(len(accs_across_folds[0:4]), training_mean_acc, f'Val: {training_mean_acc:.2%}',
        ha='right', va='top', color='red', fontsize=10)

    plt.title('Accuracy Comparison')
    plt.xlabel('Folds')
    plt.ylabel('Accuracy')
    plt.ylim(limit1, limit2)    # scope of the 'y' axis
    plt.legend()    # displays what each line represents

    plt.show()

    return test_accuracy