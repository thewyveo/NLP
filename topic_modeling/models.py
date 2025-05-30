# importations
from topic_modeling.logisticreg import logreg
from topic_modeling.lda_lsa import lda, lsa
from topic_modeling.preprocessing import preprocess
from topic_modeling.plotting import plotting

# initializing training dataset (might take a while)

def model_LogisticRegression(training_dataset):
    '''Logistic Regression Model for Topic Modeling.'''
    print('Logistic Regression Model for Topic Modeling:\n')
    (training_mean_acc, fold_accs), (y_pred, y_true), sentences = logreg(training_dataset)
    plotting((training_mean_acc, fold_accs), (y_pred, y_true), sentences)

def model_LDA(training_dataset):
    '''Latent Dirichlet Allocation (LDA) Model for Topic Modeling.'''
    print('\nSupervised Latent Dirichlet Allocation (LDA) Model for Topic Modeling:\n')
    print('=== TRAINING ===')       # training phase (takes longer than log. reg. model, couple minutes)
    (training_mean_acc_lda, history_lda), (y_pred_lda, y_true_lda), sentences, training_report = lda(training_dataset)
    print(f'\n{training_report}')

    print('\n=== TESTING ===')      # testing phase & results
    plotting((training_mean_acc_lda, history_lda), (y_pred_lda, y_true_lda), sentences)
        
def model_LSA(training_dataset):
    '''Latent Semantic Analysis (LSA) Model for Topic Modeling.'''
    print('\nSupervised Latent Semantic Analysis (LSA) Model for Topic Modeling:\n')
    (training_mean_acc_lsa, history_lsa), (y_pred_lsa, y_true_lsa), sentences, training_report = lsa(training_dataset)
    print('\nTraining Classification Report:\n', training_report)
    
    plotting((training_mean_acc_lsa, history_lsa), (y_pred_lsa, y_true_lsa), sentences)


def Topic_Modeling_Component():
    '''
    Topic Modeling Component for NLP Pipeline.
    
    This function initializes the training dataset and runs the topic modeling models.
    '''
    print('\nTopic Modeling Component:')
    print('Preprocessing topic modeling training dataset...')
    training_dataset = preprocess()  # Initialize the training dataset (might take a while)
    print('Preprocessing complete, moving onto topic modeling models\n')
    model_LogisticRegression(training_dataset)
    model_LDA(training_dataset)
    model_LSA(training_dataset)


#Topic_Modeling_Component()