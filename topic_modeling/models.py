# importations
from topic_modeling.logisticreg import logreg
from topic_modeling.lda_lsa import lda, lsa
from topic_modeling.preprocessing import preprocess
from topic_modeling.plotting import plotting
from pyfiglet import figlet_format

# initializing training dataset (might take a while)

def model_LogisticRegression(training_dataset):
    '''Logistic Regression Model for Topic Modeling.'''
    print('=== TM - Logistic Regression Model ===')
    (training_mean_acc, fold_accs), (y_pred, y_true), sentences = logreg(training_dataset)
    plotting((training_mean_acc, fold_accs), (y_pred, y_true), sentences)

def model_LDA(training_dataset):
    '''Latent Dirichlet Allocation (LDA) Model for Topic Modeling.'''
    print('\n=== TM - Supervised Latent Dirichlet Allocation (LDA) Model ===')
    print('TM - Training LDA...')       # training phase (takes longer than log. reg. model, couple minutes)
    (training_mean_acc_lda, history_lda), (y_pred_lda, y_true_lda), sentences, training_report = lda(training_dataset)
    print(f'\n{training_report}')

    print('\nTM - Evaluating LDA on test set...')      # testing phase & results
    plotting((training_mean_acc_lda, history_lda), (y_pred_lda, y_true_lda), sentences)
        
def model_LSA(training_dataset):
    '''Latent Semantic Analysis (LSA) Model for Topic Modeling.'''
    print('\n=== TM - Supervised Latent Semantic Analysis (LSA) Model ===\n')
    (training_mean_acc_lsa, history_lsa), (y_pred_lsa, y_true_lsa), sentences, training_report = lsa(training_dataset)
    print('\nTM - Training Classification Report:\n', training_report)
    
    plotting((training_mean_acc_lsa, history_lsa), (y_pred_lsa, y_true_lsa), sentences)


def Topic_Modeling_Component():
    '''
    Topic Modeling Component for NLP Pipeline.
    
    This function initializes the training dataset and runs the topic modeling models.
    '''
    print()
    print(figlet_format("Topic Modeling", font="small"))
    print('TM - Preprocessing topic modeling training dataset...')
    try:
        training_dataset = preprocess()  # Initialize the training dataset (might take a while)
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