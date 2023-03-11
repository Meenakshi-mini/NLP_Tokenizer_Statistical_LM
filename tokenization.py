# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:04:20 2023

@author: sirig
"""
import numpy as np
import pandas as pd
import re
import argparse



def tokenization(text,output_path):
  tokenized_data = []
  for i in range(0,len(text)):
    text[i] = re.split("[' ,\n\"]",text[i]) # Splitting the text into separate tokens if there is apostrophe, space, comma, new line or double quotes  
    text[i] = [w.lower() for w in text[i]] #Convert entite text to lower case
    if i==0:  # For the very first line inserting sentence start tag
      text[i].insert(0,'<s>')
    for j in range(0,len(text[i])):
      text[i][j]=re.sub(r"(https?:(//)|www\.)[a-zA-Z]\S*[a-zA-Z]\.[a-zA-Z]\S*","<url>",text[i][j])  # URL tag
      text[i][j]=re.sub(r"#","<hashtag>",text[i][j])  # Hash tag
      text[i][j]=re.sub(r"@","<mention>",text[i][j]) # Mention tag   
      if len(text[i][j])!=0:  #to avoid spaces
        if text[i][j] not in ['no.','mrs.','mr.'] and text[i][j][-1] in [".","?","!",";"]: # if the token is not in ['no.','mrs.','mr.'] and its last character is in [".","?","!",";"] will add end of the sentence token.
          out = re.sub(r'[^a-zA-Z]','', text[i][j]) #Other than characters if the token has any other special symbols will remove those symbols - Punctuations
          if len(out)==1 and out in ['a','i']:  #if the token length is 1 and they belong to 'a' or 'i' then we consider, otherwise continue with next token
            tokenized_data.append(out)
          elif len(out)>1:    #if the token length greater than 1
              tokenized_data.append(out)
          tokenized_data.append('</s>')   #adding end of the sentence tag
          tokenized_data.append('<s>')    #adding start of the sentence tag
          continue        
        if len(text[i][j])!=0 and text[i][j] not in ['<s>','</s>','<url>','<hashtag>','<mention>']:    #if token is not empty and token is not in['<s>','</s>','<url>','<hashtag>','<mention>']
          out = re.sub(r'[^a-zA-Z]','', text[i][j])    #if inbetween characters of token has any special symbols will remove those symbols - Punctuations
          if len(out)==1 and out in ['a','i']: #if the token length is 1 and they belong to 'a' or 'i' then we consider
            tokenized_data.append(out)
          elif len(out)>1:              #if the token length greater than 1
              tokenized_data.append(out)
        elif len(text[i][j])!=0 and text[i][j] in ['<s>','</s>','<url>','<hashtag>','<mention>']:  #if token is not empty and token is in['<s>','</s>','<url>','<hashtag>','<mention>']
            tokenized_data.append(text[i][j])
            
  outFile = open(output_path, "w",encoding="utf8")         
  sen = []  # To store 1 sentence at a time
  no_of_sen = 0 #to calculate no of sentences in the given corpus
  sentence_level_list = [] # To store all sentences
  for i in range(0,len(tokenized_data)): # Iterating over all the tokenized data
    sen.append(tokenized_data[i])  #appending each token to sen
    if tokenized_data[i]=="</s>":
        if len(sen)>3:  #When we encounter end of the sentence tag will append till that time stored tokens as a single sentence into the file
            outFile.write(" ".join(sen)+"\n")
            no_of_sen+=1
            sentence_level_list.append(" ".join(sen)) #combining individual tokens with space to form a sentence
        sen = [] # Emptying the contents of 'sen' to start storing the next sentence tokens in the next iteration
  print("Total no. of Sentences = ",no_of_sen)

if __name__ == "__main__":
    # Command Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    # Corpus path
    corpus_path = parser.parse_args().input_path   
    # Tokenized text Output path 
    output_path = parser.parse_args().output_path
    # Loading the corpus
    with open(corpus_path, 'r') as fp:
        texts = fp.readlines()
    tokenization(texts,output_path)  #Calling the tokenization function
