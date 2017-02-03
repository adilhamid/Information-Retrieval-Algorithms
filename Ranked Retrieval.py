
# Builiding the Vector Space Model Without saving the double point variables in the dictionary to save the memory

# Build the vector space index Model here

import os
import string
import re
import codecs
import collections
import math
import timeit
from nltk.stem.porter import *


# The Function to build the index of the data for retrieval
'''Inputs - databasePath - raw input database path :
            p_use_casefolding flag
            p_use_stemming flag
    Output - dictionary, vectorTable, vectorSquareSum for normalization''' 
def makeVectorSpaceIndexFast(dictionary, vectorTable, databasePath, p_use_casefolding, p_use_stemming):
    print 'Generating the Vector Space Model............ Please Wait.'
    
    #Intiliazation of the different parts used
    stemmer = PorterStemmer();
    countFiles =0;
    
    # Getting the list of documents we have in the database: 
    # Since we have been given the folder, I am treating all the files as documnets
    
    documentIdList =[ids for ids in os.listdir(databasePath)]   # for now using the filename as id "101.txt" 
   
    # Since we are building the term- frequency, we need the docIds in some systematic way
    documentIdList.sort()
    
    # Traversing all the documnets one by one
    vectorSquareSum = collections.defaultdict(lambda: 0.0)
    
    for id in documentIdList:
       
        with open(databasePath+id, 'r') as fileHandler:
            countFiles += 1
            outFile = fileHandler.read();
            # Close the file to read the next file
            fileHandler.close()
            
            # Using the Tokenizer function of part One
            # False means we need all the tokens as we have to calculate the term freq also
            myTokens = returnTokens(outFile,p_use_casefolding,p_use_stemming,False) 

            # Now for each term we get in the document we will try to 
            
            for term in myTokens:
                #Add to the dictionary if not already present
                dictionary.setdefault(term,[]).append(id)
        
    
    # Building the frequency of each Term and correspnding tf-Idf values as given by the Question
    for terms, docPostings in dictionary.items():
        documentFreqTable = collections.Counter(docPostings)
        docFreqTerm = len(documentFreqTable)
        
        
        for docId, termFreq in sorted(documentFreqTable.items()):
            logWFreq =0;
            if(termFreq >0):
                logWFreq = 1 + math.log10(termFreq) # Log to base 10 used in the book as checked with example 6.8
                
            weightLimit = math.log10(countFiles/float(docFreqTerm))
            vectorSquareSum[docId] += math.pow((logWFreq * weightLimit), 2)
        
        # Instead of Saving all the floating point values, just storing the integer values        
        dictTDIDF =[]
        dictTDIDF.append(docFreqTerm)
        dictTDIDF.append(dict(documentFreqTable))
        #So Vector table is dictionary of terms containing list having two values 1) document Frequecny to calc Idft 
        # 2) the dictionary of docId and term frequency in each document.
        vectorTable[terms] = dictTDIDF
        
    print 'Vector Space Model Generated'
    return vectorSquareSum
        
start = timeit.default_timer() 

# Run this cell to create the Vector Space Model
dictionary = {}
vectorSpaceModel ={}
databasePath = './southpark_scripts/'
vectorSquareSum = makeVectorSpaceIndexFast(dictionary,vectorSpaceModel,databasePath, use_casefolding, use_stemming)

stop = timeit.default_timer()                     
print ('Vector Space Build Time:' + str(stop - start)) 
    


# search for the input and print the top 5 document ids along with their associated cosine scores.

import operator

def rankedSearch(queryTerm,p_use_casefolding , p_use_stemming):
    
    stemmer = PorterStemmer();
    
    queryList = (returnTokens(queryTerm, p_use_casefolding , p_use_stemming, False)) #getting the tokens
    #print queryList
    finalScores = collections.defaultdict(lambda: 0.0)
    totalDocs = len(vectorSquareSum)
        
    # Raw term frequency for each query term
    termWeight = {}
    qVector = 0;
    for term in queryList:
        if (term in termWeight):
            termWeight[term] +=1
        else:
            termWeight[term] =1
    
    for item in termWeight:
        qVector += math.pow(termWeight[item],2);
    
    qVector = math.sqrt(qVector)   #queryVector Magnitude for normalizing the scores  
   
    for term, wtq in termWeight.items():
        # for each document listing in the vector table we will check the terms score tf-IDF * tf(of query term)
        if(term in vectorSpaceModel):
            for docId,tf in vectorSpaceModel[term][1].items():
                df = vectorSpaceModel[term][0]
                finalScores[docId] += (math.log10(totalDocs/float(df))) * (1 + math.log10(tf)) * wtq
    
    # Normalizing the scores
    for docID, score in finalScores.items():
        val = math.sqrt(vectorSquareSum[docID])
        finalScores[docID] /= val * qVector
   
    # Sorting the lists according to the scores
    sortedScores = sorted(finalScores.items(), key=operator.itemgetter(1), reverse=True) #next time we can use maxheap
    if (len(sortedScores) <= 5):
        return sortedScores
    else:
        return sortedScores[:5]

# Please Run the above cell to generate the Vector Space Model
search_text = raw_input('Ranked Search:')
start = timeit.default_timer() 
 
finalResults = rankedSearch(search_text,use_casefolding, use_stemming)

print 'Results ','--->', 'Scores'
print

if(len(finalResults) ==0 or finalResults[0][1] == 0.0):
    print 'No Results Found'
else:
    for items in finalResults:
        print items[0] ,'--->', items[1]
stop = timeit.default_timer()  
print
print ('Search Time:' + str(stop - start))