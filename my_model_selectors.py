import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose
        self.num_features = len(self.X[0])
        self.num_datapoints = len(self.X)

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN

    - Let: 
    m = # of states 
    f = # of features

    - Number of parameters are the amount of variables that HMM has to learn to optimize the model.
    - Which equals to # of probabilities in transitional matrix + # of emission probabilities + # of starting probabilities
    - Number of transitional probabilities =  # of cells in the matrix by default (m^2), but since the last column can be inferred from the rest (total probabilities has to be equal to 1), we only need m*(m-1).
    - Number of emission probabilities = 2*f*m (1 for mean and 1 for variance, assuming a diagonal covariance matrix)
    - Number of starting probabilities = m-1 (m by default but again, the last one can be inferred)

    => p = m*(m-1) + 2*f*m + (m-1) = m^2 + 2*f*m -1

    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        best_score = float("inf")
        best_model = self.base_model(self.n_constant)

        # TODO implement model selection based on BIC scores
        # Need to use try catch to avoid this error 
        # https://discussions.udacity.com/t/hmmlearn-valueerror-rows-of-transmat--must-sum-to-1-0/229995
        try:
            for num_states in range(self.min_n_components, self.max_n_components+1):
                current_model = self.base_model(num_states)
                logL = current_model.score(self.X, self.lengths)
                logN = np.log(self.num_datapoints)
                p = num_states**2 + 2*self.num_features*num_states - 1
                bic_score = -2*logL + p*logN
                if (bic_score < best_score):
                    best_score = bic_score
                    best_model = current_model
            return best_model
        except:
            return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float("-inf")
        best_model = self.base_model(self.n_constant)

        # TODO implement model selection based on DIC scores
        # DIC = (likelihood of current word) - (average likelihood of other words)
        try:
            for num_states in range(self.min_n_components, self.max_n_components+1):
                current_model = self.base_model(num_states)
                dic_score = current_model.score(self.X, self.lengths) - np.mean(
                    [current_model.score(X, lengths) for word, (X, lengths) in self.hwords.items() if word != self.this_word])
                if (dic_score > best_score):
                    best_score = dic_score
                    best_model = current_model
            return best_model
        except:
            return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        
        best_score = float("-inf")
        best_model = self.base_model(self.n_constant)

        # Can't split to K-fold with less than 3 examples
        if len(self.sequences)<3:
            return best_model

        try:
            for num_states in range(self.min_n_components, self.max_n_components+1):
                kf = KFold(3, shuffle = True)
                cv_scores = []
                # Selection criteria is the average log likehood between folds
                # To calculate average, need to loop to get score for each fold then normalize
                for train_indices, test_indices in kf.split(self.sequences):
                    train_X, train_lengths = combine_sequences(train_indices, self.sequences)
                    test_X, test_lengths = combine_sequences(test_indices, self.sequences)
                    current_model = self.base_model(num_states, train_X, train_lengths)
                    cv_scores.append(current_model.score(test_X, test_lengths))
                mean_cv_score = np.mean(cv_scores)
                if (mean_cv_score > best_score):
                    best_score = mean_cv_score
                    best_model = current_model
            return best_model
        except:
            return best_model
            