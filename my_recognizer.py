import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    
    # For each test case: 
    # - Iterate through every trained model of the words in the training set (there is 1 optimal model for each word)
    # - Make a dictionary to record the score of the test case for each model
    # - Add that dictionary to the list of probabilities
    # - Model of highest score is the final guess for that test case. Add that to the list of guesses.
    for test_word, (X, lengths) in test_set.get_all_Xlengths().items():
      test_word_probabilities = {}
      best_guess = None
      best_score = float("-inf")
      for word, model in models.items():
        try:
          score = model.score(X,lengths)
        except:
          score = float("-inf")
        test_word_probabilities[word] = score
        if score > best_score:
          best_score = score
          best_guess = word
      probabilities.append(test_word_probabilities)
      guesses.append(best_guess)

    return probabilities, guesses