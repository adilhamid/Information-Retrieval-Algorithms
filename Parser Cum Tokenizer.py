

# configuration options
use_casefolding = True # or false
use_stemming = True # or false

print 'Global Variable/Flags Set'
print use_casefolding, use_stemming


# In[13]:

# Your parser function here. It will take the two option variables above as the parameters
# add cells as needed to organize your code
import os
import string
import re
import timeit
from nltk.stem.porter import *

# Global Variables
stringPunct = re.compile('[%s]' % re.escape(string.punctuation))

# To get the Tokens using the string.punctuation
def getTokensStringPunct(str):
    return "".join(stringPunct.sub('', str))

def getTokensAlphaNum(str):
    return ''.join((char if char.isalnum() else ' ') for char in str).split()

def getTokensRegex(str):
    return re.split(r'\W+',str)
 
# Initializing the stemmer to the porter stemmer

stemmer = PorterStemmer()
cache = {}

def getFromCache(token):
    if token not in cache:
        cache[token] = stemmer.stem(token)        
    return cache[token]

# Return Tokens given a text string
'''Input -> textStr --> String to be Tokenized
            p_use_casefolding - >Whether to be casefolded or not
            p_use_stemming -> whether to use the stemming or not
            setOp -> whether to keep the duplicate tokens or not
            
    Output -> Returns to Tokens in a list format '''

def returnTokens(textStr,p_use_casefolding , p_use_stemming, setsOp):
    
#   myTokens = getTokensAlphaNum(textStr)   # Using the AlphaNum function of the string to get the tokens
    myTokens = getTokensRegex(textStr)      # Using the Regex to get the tokens
    
    if(setsOp):
        myTokens = set(myTokens)
   
    # now changing the case according to the flags given
    if p_use_casefolding:
        myTokens = [element.lower() for element in myTokens]
        if(setsOp):
            myTokens = set(myTokens)

    if p_use_stemming:
        myTokens = [getFromCache(plural) for plural in myTokens]
        if(setsOp):
            myTokens = set(myTokens)
    
    # Filter the empty strings
    myTokens = list(filter(None, myTokens))
    return myTokens


def parserFunction(path, use_casefolding, use_stemming):
    
    myTokens= set()
    outFile=""
    for filename in os.listdir(path):
        with open(path+filename, 'r') as f:
            outFile += f.read()
            f.close()
            
    myTokens = returnTokens(outFile, use_casefolding,use_stemming, True)
    
    print len(myTokens)
    
databasePath = './southpark_scripts/'   # Relative path for the folder containing the scripts

start = timeit.default_timer() 

parserFunction(databasePath,use_casefolding,use_stemming) # Using the use_casefolding and use_stemming ( you can set true false here also)

stop = timeit.default_timer()                     
print ('Index Timing:' + str(stop - start)) 
