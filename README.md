# NLP_Tokenizer_Statistical_LM_with_Smoothing


`1.  Tokenization (tokenization.py):`
The following scenarios are handled in my code:

* Word Tokenizer using ‘{space, apostrophe, space, comma, new line or double quotes}’
* URLs substituted with ‘<url>’ tag using regular expression of the form
"(https?:(//)|www\.)[a-zA-Z]\S*[a-zA-Z]\.[a-zA-Z]\S*"
* Hashtags substituted with ‘<hashtag>’ tag using regular expression of the form "#"
* Mentions substituted with ‘<mention>’ tag using regular expression of the form "@"
* Percentages, age values, expressions indicating time and time periods are
removed using regular expression of the form “[^a-zA-Z]”
* End of the sentence identified if the token is not in ['no.','mrs.','mr.'] and its last
character is in [".","?","!",";"]
* Start of the sentence tag is added after encountering ‘end of the sentence’ using the
above condition and for the very first line of the file by default start of the sentence tag is
added.
* Punctuations are removed using regular expression of the form [^a-zA-Z]

`2.  Language Model (LM.py):` (tokenization + 4-gram LM + Kneser-Ney smoothing)

  Below are the functions that are used:
  * givenN_to_unigram_frequencyGenerator(n, texts):
  The input to this function are n and texts, where ‘n’ is the max n-gram that
  we are interested in and ‘texts’ is the training corpus for which we want to buildn-grams. This function returns a dictionary consisting of each unigram to
  mentioned ‘n’ gram frequency count possible sequence in training corpus,
  *  given_Ngram_frequency(ngram, ngrams_dict):
  The input to this function is ngram and ngrams_dict, where ngram is the
  string that we are interested in finding its frequency count and ngrams_dict
  contains frequency count of all possible ngrams where n varies from 1 to 4(in this
  case). The output is the frequency of a given ngram.
  * given_Ngram_unique_occurence_count(ngram, ngrams_dict):
  The input to this function is ngram and ngrams_dict, where ngram is the
  string that we are interested in finding its count of its occurrence with different
  word context and ngrams_dict contains frequency count of all possible ngrams
  where n varies from 1 to 4(in this case). The output is unique occurrence of given
  ngrams with different word context count.
  * kneserNey_Smoothing(history, currentword, recur_step, ngrams_dict):
  This function performs a task similar to below formulation and ‘d’ i took as
  ‘0.75’.
  * wittenBell_Smoothing(history, current, ngrams_dict):
  This function performs a task similar to below formulation.
  * calculatePP(sentence, n, smoothing, ngrams_dict):
  For a given sentence, it calculates the scores by applying a given
  smoothing technique.
