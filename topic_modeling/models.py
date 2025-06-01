from topic_modeling.logisticreg import logreg
from topic_modeling.lda_lsa import lda, lsa
from topic_modeling.preprocessing import preprocess
from topic_modeling.plotting import plotting
from pyfiglet import figlet_format


def model_LogisticRegression(training_dataset):
    '''
    Logistic Regression model for Topic Modeling. Trains a Logistic Regression model
    on the preprocessed training dataset, evaluates it on the test dataset, and plots the results.

    :input: training_dataset: the preprocessed training dataset
    '''

    print('=== TM - Logistic Regression Model ===')
    (training_mean_acc, fold_accs), (y_pred, y_true), sentences = logreg(training_dataset)
    plotting((training_mean_acc, fold_accs), (y_pred, y_true), sentences)

def model_LDA(training_dataset):
    '''
    Latent Dirichlet Allocation (LDA) model for Topic Modeling. Trains an LDA model
    on the preprocessed training dataset, evaluates it on the test dataset, and plots the results.

    :input: training_dataset: the preprocessed training dataset
    '''

    print('\n=== TM - Supervised Latent Dirichlet Allocation (LDA) Model ===')
    print('TM - Training LDA...')       # training phase (takes longer than log. reg. model, couple minutes)
    (training_mean_acc_lda, history_lda), (y_pred_lda, y_true_lda), sentences, training_report = lda(training_dataset)
    print(f'\n{training_report}')

    print('\nTM - Evaluating LDA on test set...')      # testing phase & results
    plotting((training_mean_acc_lda, history_lda), (y_pred_lda, y_true_lda), sentences)
        
def model_LSA(training_dataset):
    '''
    Latent Semantic Analysis (LSA) model for Topic Modeling. Trains an LSA model
    on the preprocessed training dataset, evaluates it on the test dataset, and plots the results.
    
    :input: training_dataset: the preprocessed training dataset
    '''

    print('\n=== TM - Supervised Latent Semantic Analysis (LSA) Model ===\n')
    (training_mean_acc_lsa, history_lsa), (y_pred_lsa, y_true_lsa), sentences, training_report = lsa(training_dataset)
    print('\nTM - Training Classification Report:\n', training_report)
    
    plotting((training_mean_acc_lsa, history_lsa), (y_pred_lsa, y_true_lsa), sentences)


def Topic_Modeling_Component():
    '''
    Main function to run the Topic Modeling component. Preprocesses the training data,,
    trains three models (Logistic Regression, LDA, LSA) on the training data, and evaluates
    the models on the test data. Finally the results and analyses are printed. (the evaluation
    on the test data and results/analyses are done within different modules, and called within
    the models themselves. this function is just the main entry point for the Topic Modeling component)
    '''

    print()
    print(figlet_format("Topic Modeling", font="small"))    ## i thought this ascii title was a nice touch -k
    print('TM - Preprocessing topic modeling training dataset...')
    try:
        training_dataset = preprocess()  # initialize the training dataset (might take a while)
    except Exception as e:
        print(f'TM - Error during preprocessing: {e}.')
        print('TM - This is most likely due to the dataset not being downloaded or being in the wrong directory.' \
        'Please refer to the README.md file.')
        return
    print('TM - Preprocessing complete, moving onto topic modeling models\n')
    model_LogisticRegression(training_dataset)
    model_LDA(training_dataset)
    model_LSA(training_dataset)
    

#Topic_Modeling_Component()