
# Building the Index : Please Treat Each cell as the modular Code

import os
import string
import re
import codecs
import timeit
from nltk.stem.porter import *


# The Function to build the index of the data for retrieval
'''Inputs - databasePath - raw input database path 
   dictionary ->  to be built'''

def makeBinaryIndex(dictionary, databasePath, p_use_casefolding, p_use_stemming):
    #Intiliazation of the different parts used
    stemmer = PorterStemmer();
    
    countFiles = 0
    
    # Getting the list of documents we have in the database: 
    #Since we have been given the folder, I am treating all the files as documnets
    
    documentIdList =[ids for ids in os.listdir(databasePath)]   # for now using the filename as id "101.txt" 
    
    # Traversing all the documnets one by one
    for id in documentIdList:
       
        with open(databasePath+id, 'r') as fileHandler:
            countFiles += 1
            outFile = fileHandler.read();
           
            
            # Using the Tokenizer function of part One            
            myTokens = set(returnTokens(outFile, p_use_casefolding,p_use_stemming, True))
            
            #dictionary.update(dict.fromkeys(myTokens,id))    # Using the inbuilt function to merge but take more time & Space

            # Now for each term we get in the document we will try to 
            for term in myTokens:
                #Add to the dictionary if not already present
                if(term not in dictionary):
                    dictionary[term] = [id]
                else:
                    if (dictionary[term][-1] != id):
                        dictionary[term].append(id)
        # Close the file to read the next file
        fileHandler.close()
            
        
    print 'Total Files Traversed: ', countFiles


    
invertedIndex = {}
start = timeit.default_timer() 
print 'Generating the Search Index Files............ Please Wait.'

databasePath = './southpark_scripts/'
makeBinaryIndex(invertedIndex,databasePath, use_casefolding, use_stemming)

print 'Search Index Generated'
print 'Length of Dictionary:', len(invertedIndex)

stop = timeit.default_timer()                     
print ('Index Timing:' + str(stop - start)), 'seconds' 


# Please run this before going for searching 

# Processing the query for search purpose Just Implementing the AND boolean retrieval here

def binaryQuerySearch(invertedIndex,queryTerms):
    
    stemmer = PorterStemmer();
    
    queryList = (returnTokens(queryTerms, use_casefolding,use_stemming, True)) #
    
    #print queryList
    
    resultList=[]  # to save all the lists of the inverted index for each query term
    
    finalResult = []
    
    for term in queryList:
        if(term in invertedIndex):
            resultList.append(invertedIndex[term]) #Getting the results from the dictionary we already created.
        else:
            return finalResult              #We know that there is a term with zero postings that means total and is going to be 0
        
    resultList.sort() # Sorting the list according to the length of the postings to reduce the number of AND operations
    
    if(len(resultList) > 0):
        
        #I could have also used the set & operation: but since i already implemented it and also i am learning python so i went through with my code
        if(len(resultList[0]) == 0):
            return finalResult;
        
        
        finalResult.append(resultList.pop(0)) # Populating the prevAndedResult with first element of the Result List
        
        while resultList:
            queList = resultList.pop(0)
            prevAndResult = finalResult.pop()

            indexOp1 = 0
            indexOp2 = 0
            tempList = []
            
            while(indexOp1 < len(queList) and indexOp2 < len(prevAndResult)):
                #Checking the three cases of the matching the document ids of the query terms

                if(queList[indexOp1] == prevAndResult[indexOp2]):
                    tempList.append(queList[indexOp1])
                    indexOp1 +=1
                    indexOp2 +=1

                elif(queList[indexOp1] > prevAndResult[indexOp2]):   # Optimization needed for incrementing by large amount
                    indexOp2 +=1

                else:
                    indexOp1 +=1
            
            finalResult.append(tempList)
    
        return finalResult
    


search_text = raw_input('Boolean Search:')

start = timeit.default_timer() 
print 'Searching the query .......... Please Wait.'

# please Bild the invertedIndex in the above Cell first
result = binaryQuerySearch(invertedIndex,search_text)  

if(result is None or len(result) != 1): 
    print 'No Results Found from the Search Engine'
else: 
    print 'Results Found'
    print len(result[0])
    print result

stop = timeit.default_timer()                     
print ('Search Timing:' + str(stop - start)) 
