'''
Submission by: Roey Elmelech
UNI: re2533
'''

import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Spring 2023 
Programming Homework 1 - Trigram Language Models
Instructor: Prof. Daniel Bauer
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
    
    elif n == 1:
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
        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)
        self.total_word_count = 0

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
        # EDGE CASE 1: if first word(s) is/are START
        if trigram[:2] == ('START', 'START',) or trigram[:1] == ('START',):
            return self.raw_bigram_probability(trigram[1:])
        
        # EDGE CASE 2: if trigram not in dictionary
        elif trigram not in self.trigramcounts:
            return 1 / self.total_word_count
        
        # EDGE CASE 3: if denominator is 0
        elif self.bigramcounts[trigram[:2]] == 0:
            return 1 / self.total_word_count

        return self.trigramcounts[trigram] / self.bigramcounts[trigram[:2]]

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        # EDGE CASE 1: if first word is START
        if bigram[:1] == ('START',):
            return self.raw_unigram_probability(bigram[1:])
        
        # EDGE CASE 2: if bigram not in dictionary
        elif bigram not in self.bigramcounts:
            return 1 / self.total_word_count
        
        # EDGE CASE 3: if denominator is 0
        elif self.unigramcounts[bigram[:1]] == 0:
            return 1 / self.total_word_count

        return self.bigramcounts[bigram] / self.unigramcounts[bigram[:1]]
        
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
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
        lambda_value = 1/3.0

        return (lambda_value * self.raw_trigram_probability(trigram) + lambda_value * self.raw_bigram_probability(trigram[1:]) + lambda_value * self.raw_unigram_probability(trigram[2:]))
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigram = get_ngrams(sentence, 3)
        log_prob = 0.0
        for sequence in trigram:
            log_prob += math.log2(self.smoothed_trigram_probability(sequence))
        
        return log_prob
    
    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        M = 0
        sum_of_logs = 0
        for sentence in corpus:
            sum_of_logs += self.sentence_logprob(sentence)
            # +1 for the STOP token
            M += len(sentence) + 1
        return 2 ** -(sum_of_logs / M)

def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            comp = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if pp < comp:
                correct += 1

            total += 1
    
        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            comp = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            if pp < comp:
                correct += 1

            total += 1

        return correct / total
        
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
    # print("Testing perplexity: ")
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment: 
    # print("Essay scoring experiment: ")
    # acc = essay_scoring_experiment("hw1_data/ets_toefl_data/train_high.txt", "hw1_data/ets_toefl_data/train_low.txt", "hw1_data/ets_toefl_data/test_high", "hw1_data/ets_toefl_data/test_low")
    # print("{:.2f}".format(acc * 100) , "%")

    # print(get_ngrams(["natural","language","processing"],3)) # Part 1
    # print(model.trigramcounts[('START','START','the')]) # Part 2
    # print(model.bigramcounts[('START','the')]) # Part 2
    # print(model.unigramcounts[('the',)]) # Part 2
    # print("Raw Trigram Probability for ('START','START','the'): ") # Part 3
    # print("{:.15f}".format(float(model.raw_trigram_probability(('START','START','the'))))) # Part 3
    # print("Raw Bigram Probability for ('START','the'): ") # Part 3
    # print("{:.15f}".format(float(model.raw_bigram_probability(('START','the'))))) # Part 3
    # print("Raw Unigram Probability for ('the'): ") # Part 3
    # print("{:.15f}".format(float(model.raw_unigram_probability(('the',))))) # Part 3
    # print("Smoothed Trigram Probability for ('START','START','the'): ") # Part 4
    # print("{:.15f}".format(float(model.smoothed_trigram_probability(('START','START','the'))))) # Part 4
    # print(model.sentence_logprob(["natural","language","processing"])) # Part 5