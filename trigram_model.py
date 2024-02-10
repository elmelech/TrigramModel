import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2023 
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    
    # Check if n is valid
    if n <= 0:
        raise ValueError("Invalid value of n")
    
    elif n ==1:
        padded_sequence = ['START'] + sequence + ['STOP']

    # Pad the sequence with 'START' and 'STOP' tokens
    else:
        padded_sequence = ['START'] * (n - 1) + sequence + ['STOP']
    
    # Generate n-grams
    ngrams = [tuple(padded_sequence[i:i+n]) for i in range(len(padded_sequence) - n + 1)]
    
    return ngrams


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)
        self.total_word_count = 0

        ##Your code here - nothing is returned, but self.unigramcounts, etc. will have counts
        for sentence in corpus:
            unigrams = get_ngrams(sentence, 1)
            bigrams = get_ngrams(sentence, 2)
            trigrams = get_ngrams(sentence, 3)

            for unigram in unigrams:
                if unigram in self.unigramcounts and unigram != ('START',):
                    self.unigramcounts[unigram] += 1
                    self.total_word_count += 1
                else:
                    self.unigramcounts[unigram] = 1

            for bigram in bigrams:
                if bigram in self.bigramcounts:
                    self.bigramcounts[bigram] += 1
                else:
                    self.bigramcounts[bigram] = 1
                    
            for trigram in trigrams:
                if trigram in self.trigramcounts:
                    self.trigramcounts[trigram] += 1
                else:
                    self.trigramcounts[trigram] = 1

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        # It's expected that many of the n-gram probabilities are 0, that is the ngram sequence has not been observed during training. One issue you will encounter is the case if which you have a trigram u,w,v   where  count(u,w,v) = 0 but count(u,w) is also 0. In that case, it is not immediately clear what P(v | u,w) should be. My recommendation is to make P(v | u,w) = 1 / |V|  (where |V| is the size of the lexicon), if count(u,w) is 0. That is, if the context for a trigram is unseen, the distribution over all possible words in that context is uniform.  Another option would be to use the unigram probability for v, so P(v | u,w) = P(v). 

        # The arguments will be individual ngrams (formatted as a tuple).
        # https://edstem.org/us/courses/53508/discussion/4214983

        # EDGE CASE: if first two words are START START - return self.raw_bigram_probability(trigram[1:])
        # EDGE CASE 2: if self.trigramcounts[trigram] == 0 - return 1 / total_word_count (unifrom distribution over all possible words in that context)
        # EDGE CASE 3: if self.bigramcounts[(trigram[0], trigram[1])] == 0 - return self.raw_unigram_probability(trigram[2])

        if trigram == ('START','START'): # shouldn't this be ('START','START','START')?
            return self.raw_bigram_probability(trigram[1:])
        
        elif self.trigramcounts[trigram] == 0:
            return (1 / self.total_word_count)
        
        if self.bigramcounts[(trigram[0], trigram[1])] == 0:
            return self.raw_unigram_probability(trigram[2])

        
        return self.trigramcounts[trigram] / self.bigramcounts[(trigram[0], trigram[1])]
            

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        # EDGE CASE: if self.bigram[trigram] == 0 - return 1 / total_word_count (unifrom distribution over all possible words in that context)
        if self.unigramcounts[(bigram[0])] == 0:
            return (1 / self.total_word_count)

        return self.bigramcounts[bigram] / self.unigramcounts[(bigram[0])]
        
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        
        # if first word is START - return 0
        if unigram == ('START',):
            return 0
        
        return self.unigramcounts[unigram] / self.total_word_count
        
    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        # see formula in the slides or in https://edstem.org/us/courses/53508/discussion/4208569
        return (lambda1 * self.raw_trigram_probability(trigram) + lambda2 * self.raw_bigram_probability(trigram[1:]) + lambda3 * self.raw_unigram_probability(trigram[2:]))
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        # What is the datatype of "sentence" in sentence_logprob(sentence)? Can we assume it will be given as an array? Or is it a string? -It’s a sequence (either a list or tuple) of tokens. https://edstem.org/us/courses/53508/discussion/4215697
        return float("-inf")

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        return float("inf") 


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            # .. 
    
        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            # .. 
        
        return 0.0

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment: 
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    # print(acc)

    # print(get_ngrams(["natural","language","processing"],3)) # Part 1
    # print(model.trigramcounts[('START','START','the')]) # Part 2
    # print(model.bigramcounts[('START','the')]) # Part 2
    # print(model.unigramcounts[('the',)]) # Part 2
    # print(model.raw_trigram_probability(('START','START','the'))) # Part 3
    # print(model.raw_bigram_probability(('START','the'))) # Part 3
    # print(model.raw_unigram_probability(('the',))) # Part 3
    # print(model.smoothed_trigram_probability(('START','START','the'))) # Part 4
    # print(model.sentence_logprob(["natural","language","processing"])) # Part 5

