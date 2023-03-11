import pickle
import numpy as np
import argparse


def givenN_to_unigram_frequencyGenerator(n, texts):    
    all_1_to_N_freq_dict = {}
    for k in range(1,n+1):        
        kgrams = dict()
        for sentence in texts:        
            tokens = sentence.split(' ')
            if len(tokens) > 0:        
                ngrams_list = []
                tokens_s = ["<s>"]*(k-2)+tokens
                for t in range(k, len(tokens_s)):            
                    ngrams_list.append(" ".join(tokens_s[t-k:t])) 
            else:
                ngrams_list = None
            
            if ngrams_list is not None:
                for s in ngrams_list:
                    if s not in kgrams:
                        kgrams[s] = 1
                    else:
                        kgrams[s] += 1  
        if k==1:
            unigrams = kgrams
            unigrams["<UNK>"] = 0
            for key, val in unigrams.copy().items():
                if val <= 1 and key!="<UNK>":
                    unigrams["<UNK>"] += val            
                    unigrams.pop(key)         
            all_1_to_N_freq_dict[1] = unigrams
            
        all_1_to_N_freq_dict[k] = kgrams
    return all_1_to_N_freq_dict

def given_Ngram_frequency(ngram, ngrams_dict):

    count = 0
    n = len(ngram.split(" ")) + 1
    for key, val in ngrams_dict[n].items():
        req_str = " ".join(key.split(" ")[:-1])
        if req_str == ngram:
            count += val
    return count

def given_Ngram_unique_occurence_count(ngram, ngrams_dict):

    count = 0
    n = len(ngram.split(" ")) + 1    
    for key, val in ngrams_dict[n].items():
        if val > 0:
            req_str = " ".join(key.split(" ")[:-1])
            if req_str == ngram:
                count += 1
    return count

def kneserNey_Smoothing(history, currentword, recur_step, ngrams_dict):

    n = len(history.split())+1
    if currentword not in ngrams_dict[1]:        
        return 0.75/ngrams_dict[1]["<UNK>"]

    if n == 1:
        return 0.25/len(ngrams_dict[1]) + 0.75/ngrams_dict[1]["<UNK>"]
    if recur_step == 1:
        try:
            ngram_kn = " ".join([history, currentword])
            if len(ngram_kn) == 0 or ngram_kn not in ngrams_dict[len(ngram_kn.split(" "))] :
                pKN_int = 0
            else:
                pKN_int = ngrams_dict[len(ngram_kn.split(" "))][ngram_kn]
            first_term = max(pKN_int-0.75, 0)/given_Ngram_frequency(history, ngrams_dict)
            
        except ZeroDivisionError:            
            first_term = 0
    else:
        try:
            cnt = 0
            for key in ngrams_dict[n].keys():
                if key.split(" ")[-1] == currentword:
                    cnt += 1
            first_term = max(cnt - 0.75, 0)/len(ngrams_dict[n])
        except ZeroDivisionError:
            first_term = 0

    try:
        lambdaa = (0.75/given_Ngram_frequency(history, ngrams_dict))*given_Ngram_unique_occurence_count(history, ngrams_dict)
        
    except ZeroDivisionError:
        return 0.75/ngrams_dict[1]["<UNK>"]

    new_hist = " ".join(history.split()[1:])
    sec_term = lambdaa*kneserNey_Smoothing(new_hist, currentword, recur_step+1, ngrams_dict)
    return first_term + sec_term
    

def wittenBell_Smoothing(history, current, ngrams_dict):
    
    n = len(history.split()) + 1
    if n == 1:
        if current in ngrams_dict[1]:
            ngram_wb = " ".join([history, current])
            if len(current) == 0 or current not in ngrams_dict[1] :
                pWB_int = 0
            else:
                pWB_int = ngrams_dict[1][current]
                
            return pWB_int/ngrams_dict[1]["<UNK>"]
        return 1/len(ngrams_dict[1])
    try:
        lambdaa = given_Ngram_unique_occurence_count(history, ngrams_dict)/(given_Ngram_unique_occurence_count(history, ngrams_dict) + given_Ngram_frequency(history, ngrams_dict))
    except ZeroDivisionError:
        return 1/len(ngrams_dict[n])

    ngram_wb = " ".join([history, current])
    if len(ngram_wb) == 0 or ngram_wb not in ngrams_dict[len(ngram_wb.split(" "))] :
        pWB_int = 0
    else:
        pWB_int = ngrams_dict[len(ngram_wb.split(" "))][ngram_wb]
    pWB = pWB_int/given_Ngram_frequency(history, ngrams_dict)

    new_hist = " ".join(history.split()[1:])
    return (1 - lambdaa)*pWB + lambdaa*wittenBell_Smoothing(new_hist, current, ngrams_dict)

def calculatePP(sentence, n, smoothing, ngrams_dict):
      
    tokens = sentence.split(' ')
    if len(tokens) > 0:        
        ngrams_lst = []
        tokens_s = ["<s>"]*(n-2)+tokens
        for k in range(n, len(tokens_s)):            
            ngrams_lst.append(" ".join(tokens_s[k-n:k])) 
    else:
        ngrams_lst = None
        
    scores = [] 
    if smoothing == "k" and ngrams_lst is not None:
        for ng in ngrams_lst:
            lst = ng.split(" ")
            hist, current = " ".join(lst[:-1]), lst[-1]
            scores.append(kneserNey_Smoothing(hist, current, 1, ngrams_dict))
    elif smoothing == "w" and ngrams_lst is not None:
        for ng in ngrams_lst:
            lst = ng.split(" ")
            hist, current = " ".join(lst[:-1]), lst[-1]
            scores.append(wittenBell_Smoothing(hist, current, ngrams_dict))
    else:
        raise ValueError("Invalid smoothing type, please enter w for WittenBell and k for KneserNey ")
    
    if np.prod(scores)==0 or len(scores)==0:
        return 0    
    return np.power(1/np.prod(scores), 1/len(scores))


# ======================================= DRIVER CODE ================================================
if __name__ == '__main__':

    # ----------------------- Parse the Command Line Arguments -------------------
#    parser = argparse.ArgumentParser()
#    parser.add_argument('n_value', type=int)
#    parser.add_argument('smoothing', type=str)
#    parser.add_argument('path', type=str)

    # Retrieve Arguments
    n=4
    corpus_path='/home/meenakshi/Pictures/Assignment1/UL_tokenized_data';
    smoothing='w'
    with open(corpus_path, 'r') as fp:
        tokenized_data = fp.readlines() 
    print(len(tokenized_data))

    np.random.seed(23)
    test_index = np.random.choice(len(tokenized_data), round(len(tokenized_data)*0.2), replace=False)
    training_data = []
    testing_data = []
    for id in range(len(tokenized_data)):
        if id in test_index:
            testing_data.append(tokenized_data[id])
        else:
            training_data.append(tokenized_data[id])
            
    try:       
        with open(corpus_path.split("/")[-1][:-4]+"CentralDictTrain.pickle", 'rb') as handle:
            ngrams_dict = pickle.load(handle)            
    except FileNotFoundError:
        ngrams_dict = givenN_to_unigram_frequencyGenerator(n=n, texts=training_data)
        with open(corpus_path.split("/")[-1][:-4]+"CentralDictTrain.pickle", "wb") as handle:
            pickle.dump(ngrams_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    pp_train, pp_test = [], []
    LM = input("Enter the file name to store the perplexity scores: ")
#training
    with open("2022802006_"+LM+"_train-perplexity.txt", "w") as train_file:
        for no, sentence in enumerate(training_data):
            sentence = sentence.strip()
            if len(sentence) != 0:
                pp_train.append(calculatePP(sentence, n, smoothing, ngrams_dict))
                train_file.write(sentence+"    PP Score = {0:.3f}\n".format(pp_train[-1]))
                
    with open("2022802006_"+LM+"_train-perplexity.txt", 'r+') as train_file:
        sen = train_file.read()
        train_file.seek(0, 0)
        train_file.write("Average Perplexity Score: {0:.3f}\n".format(np.mean(pp_train)) + sen)

#test
    with open("2022802006_"+LM+"_test-perplexity.txt", "w") as test_file:
        for no, sentence in enumerate(testing_data):
            sentence = sentence.strip()
            if len(sentence) != 0:
                pp_test.append(calculatePP(sentence, n, smoothing, ngrams_dict))
                test_file.write(sentence+"    PP Score = {0:.3f}\n".format(pp_test[-1]))                
    with open("2022802006_"+LM+"_test-perplexity.txt", 'r+') as fll:
        sen = test_file.read()
        test_file.seek(0, 0)
        test_file.write("Average Perplexity Score: {0:.3f}\n".format(np.mean(pp_test)) + sen)


