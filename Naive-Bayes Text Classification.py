

# Parsing the Training and Testing Data to Store only the Features and Labels
import os
import string
import re
import timeit
import json
import math
from pprint import pprint
from collections import Counter


def returnTokens(textStr):
    
    myTokens = re.split(r'\W+',textStr)     
   
    myTokens = [element.lower() for element in myTokens]
        
    myTokens = list(filter(None, myTokens))
    return myTokens

def parserFunction(filename, Relevant_Data,Irrelevant_Data):
    count_Rel_Docs = 0;
    count_Irrel_Docs = 0;
    for line in open(filename, 'r'): # Reading each Revieiw one by one
        reviews = json.loads(line)

        temp_vocab = returnTokens(reviews['text'])
        
        if reviews['label'] == "Food-irrelevant":
            count_Irrel_Docs += 1
            Irrelevant_Data.extend(temp_vocab)
        else:
            count_Rel_Docs += 1
            Relevant_Data.extend(temp_vocab)

    return [count_Rel_Docs,count_Irrel_Docs]


# Build the naive bayes classifier

def TrainMultinomialNB(classRelVocab, classIrrVocab, lambda_val ):
    # To Store the Conditional Probability of each token 
    DocumentCollection = []
    DocumentCollection.extend(classRelVocab)
    DocumentCollection.extend(classIrrVocab)
    TrainingVocab = set(DocumentCollection)
    condProbability = dict.fromkeys(TrainingVocab)
   
    countRel = Counter(classRelVocab)
    countIrrel = Counter(classIrrVocab)
    countDocColl = Counter(DocumentCollection)

    for token in TrainingVocab:
        relvProb  = lambda_val * ((countRel[token])/(float(len(classRelVocab)))) + (1-lambda_val) *((countDocColl[token] )/(float(len(countDocColl)))) 
        irrelProb = lambda_val * ((countIrrel[token])/(float(len(classIrrVocab)))) + (1-lambda_val) *((countDocColl[token])/(float(len(countDocColl))))
        condProbability[token] = [irrelProb,relvProb]
   
    return condProbability



start = timeit.default_timer()
Relevant_Data = []
Irrelevant_Data = []
start = timeit.default_timer()
count_Rel_List = parserFunction("./training_data.json", Relevant_Data,Irrelevant_Data)

lambda_val = 0.7
condProb = TrainMultinomialNB(Relevant_Data,Irrelevant_Data, lambda_val)
lenRel = count_Rel_List[0]
lenIrel = count_Rel_List[1]

stop = timeit.default_timer() 
print ('Model Build Timing: ' + str(stop - start)) 


#NB Test

def TestNB(filename,condProb, lenRel, lenIrel):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    prior_prob_rel   = (lenRel / (float)(lenRel+lenIrel))
    prior_prob_irrel = (lenIrel / (float)(lenRel+lenIrel))
    
    score_rel = 0
    score_irrel = 0
    
    for line in open(filename, 'r'): # Reading each Revieiw one by one
        reviews = json.loads(line)

        tokens = returnTokens(reviews['text'])

        score_rel = math.log(prior_prob_rel)
        score_irrel = math.log(prior_prob_irrel)

        for token in tokens:
            if token in condProb:
                score_irrel += math.log(condProb[token][0])
                score_rel +=  math.log(condProb[token][1])                     
                
     
        ## Getting the Confusion Matrix
        if score_irrel >= score_rel:
            if reviews['label'] == "Food-irrelevant":
                TN += 1
            else:
                FN += 1
        else:
            if reviews['label'] == "Food-relevant": 
                TP += 1
            else:
                FP += 1            

    print 'Confusion Matrix'
    print TP , FP
    print FN , TN
    print 'Relevant Statistics: '
    print 'Accuracy= ', (TP+TN)/(float(TP+FP+FN+TN))
    print 'Relevant-Precesion= ' , TP / (float(TP+FP))
    print 'Relevant-Recall= ', TP/(float(TP+FN))
    
    print 
    
    print 'Irrelevant Statistics: '
    print 'Ir-Relevant-Precesion= ' , TN / (float(TN+FN))
    print 'Ir-Relevant-Recall= ', TN/(float(TN+FP))
    

# Call Testing 
start = timeit.default_timer()
print 'Started Testing'
TestNB('./testing_data.json',condProb, lenRel, lenIrel)
print 'Ended Testing'
stop = timeit.default_timer() 

print ('Testing Time Timing: ' + str(stop - start)) 



# **Results**
# 
# *Relevant Statistics:*
# 
# Accuracy=  0.9
# 
# Relevant-Precision=  0.914285714286
# 
# Relevant-Recall=  0.941176470588
# 
# 
# *Irrelevant Statistics:* 
# 
# Ir-Relevant-Precision=  0.8125
# 
# Ir-Relevant-Recall=  0.866666666667
# 

# # Part 2: Rocchio classifier [35 points]

# In this part, your job is to implement a Rocchio classifier for "food-relevant vs. food-irrelevant". You need to aggregate all the reviews of each class, and find the center. **Use the normalized raw term frequency**.
# 
# 
# ### What to report
# 
# * For the entire testing dataset, report the overall accuracy.
# * For the class "Food-relevant", report the precision and recall.
# * For the class "Food-irrelevant", report the precision and recall.
# 
# We will also grade on the quality of your code. So make sure that your code is clear and readable.

# In[6]:

# Build the Rocchio classifier
# Insert as many cells as you want

#Since we have already read the Reviews we can Reuse the same data, so before running the Rocchio Classifier, we should run the 
# Naive Bayes based clasification
def TrainRocchio(TrainingVocab,filename):
    
    cntRel = 0
    cntIrrel = 0
    mean_Relevant = dict.fromkeys(TrainingVocab, 0.0)
    mean_Irrelevant = dict.fromkeys(TrainingVocab, 0.0)
    
    # Populating the Relevant_Data_Matrix
    
    for line in open(filename, 'r'): # Reading each Revieiw one by one
        reviews = json.loads(line)        
        temp_vocab = returnTokens(reviews['text'])
        
        normalFactor = 0
        Relevant_Data_matrix = {}
        Irrelevant_Data_Matrix = {}
        
        if reviews['label'] == "Food-irrelevant":
            #Populate the Irrelevant one            
            Irrelevant_Data_Matrix =  Counter(temp_vocab)
            for tok in Irrelevant_Data_Matrix:
                normalFactor += math.pow(Irrelevant_Data_Matrix[tok],2)
            normalFactor = math.sqrt(normalFactor)
            
            #Normalizing the Values
            for tok in Irrelevant_Data_Matrix:
                Irrelevant_Data_Matrix[tok] = Irrelevant_Data_Matrix[tok]/(float(normalFactor))
                mean_Irrelevant[tok] += Irrelevant_Data_Matrix[tok]
                
            cntIrrel += 1
           
        else:
             # Populate the Relevant 
            Relevant_Data_matrix =  Counter(temp_vocab)
            for tok in Relevant_Data_matrix:
                normalFactor += math.pow(Relevant_Data_matrix[tok],2)
            normalFactor = math.sqrt(normalFactor)
            
            #Normalizing the Values
            for tok in Relevant_Data_matrix:
                Relevant_Data_matrix[tok] = Relevant_Data_matrix[tok]/(float(normalFactor))
                mean_Relevant[tok] += Relevant_Data_matrix[tok]
            
            cntRel += 1
            
    for tok, val in mean_Relevant.items():
        mean_Relevant[tok] /= cntRel
    for tok, val in mean_Irrelevant.items():
        mean_Irrelevant[tok] /= cntIrrel
    
    Mean_Classes = [mean_Irrelevant, mean_Relevant]
    
    return Mean_Classes
        

start = timeit.default_timer()
print 'Training Started....'
DocumentCollection = []
DocumentCollection.extend(Relevant_Data)
DocumentCollection.extend(Irrelevant_Data)
TrainingVocab = set(DocumentCollection)

Mean_Classes_List = TrainRocchio(TrainingVocab, './training_data.json')
print 'Training Ended . . '

stop = timeit.default_timer() 

print ('Training Time Timing: ' + str(stop - start)) 


# In[44]:

# Apply your classifier on the test data. Report the results.
# Insert as many cells as you want

def TestRocchio(filename,Mean_Classes_List, TrainingVocab, scoringMethod):
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    if scoringMethod == 2:
        normalFactor1 = 0.0
        normalFactor0 = 0.0
        for tok in TrainingVocab:
            normalFactor0 += math.pow(Mean_Classes_List[0][tok],2)
            normalFactor1 += math.pow(Mean_Classes_List[1][tok],2)
        normalFactor0 = math.sqrt(normalFactor0)
        normalFactor1 = math.sqrt(normalFactor1)

    # Test Each Entry with the Mean
    for line in open(filename, 'r'): # Reading each Revieiw one by one
        reviews = json.loads(line)        
        temp_vocab = returnTokens(reviews['text'])
        
        normalFactor = 0;        
        DataMat =  Counter(temp_vocab)
        for tok in DataMat:
            normalFactor += math.pow(DataMat[tok],2)
        normalFactor = math.sqrt(normalFactor)

        #Normalizing the Values
        for tok, freq in DataMat.items():
            DataMat[tok] = freq/(float(normalFactor))

        # Prediciting the Class        
        score = [0.0 , 0.0 ]
        
        # Euclidean Distance Based Scoring
        if scoringMethod == 1:
            totalVocab = set()
            totalVocab.update(DataMat)
            totalVocab.update(TrainingVocab)
            for tok in DataMat:
                if tok in totalVocab:
                    if tok in DataMat and tok in TrainingVocab:
                        score[0] += math.pow((DataMat[tok] - Mean_Classes_List[0][tok]),2)
                        score[1] += math.pow((DataMat[tok] - Mean_Classes_List[1][tok]),2)
                    elif tok in TrainingVocab:
                        score[0] += math.pow((Mean_Classes_List[0][tok]),2)
                        score[1] += math.pow((Mean_Classes_List[1][tok]),2) 
                    elif tok in DataMat:
                        score[0] += math.pow(DataMat[tok],2)
                        score[1] += math.pow(DataMat[tok],2) 

            score[0] = math.sqrt(score[0])
            score[1] = math.sqrt(score[1])

            if score[0] <= score[1]:
                if reviews['label'] == "Food-irrelevant":
                    TN += 1
                else:
                    FN += 1
            else:
                if reviews['label'] == "Food-relevant":
                    TP += 1
                else:
                    FP += 1
        
        #Cosine Based Score
        elif scoringMethod == 2 :
            for tok in DataMat:
                if tok in TrainingVocab:
                    score[0] += DataMat[tok] * Mean_Classes_List[0][tok]
                    score[1] += DataMat[tok] * Mean_Classes_List[1][tok]

            score[0] = score[0]/(float)(normalFactor0)
            score[1] = score[1]/(float)(normalFactor1)
        
            if score[0] >= score[1]:
                if reviews['label'] == "Food-irrelevant":
                    TN += 1
                else:
                    FN += 1
            else:
                if reviews['label'] == "Food-relevant":
                    TP += 1
                else:
                    FP += 1
        #Manhattan Based Scores
        elif scoringMethod == 3:
            for tok in TrainingVocab:
                if tok in DataMat:
                    score[0] += abs(DataMat[tok] - Mean_Classes_List[0][tok])
                    score[1] += abs(DataMat[tok] - Mean_Classes_List[1][tok])
                else:
                    score[0] += abs(Mean_Classes_List[0][tok])
                    score[1] += abs(Mean_Classes_List[1][tok]) 

            if score[0] <= score[1]:
                if reviews['label'] == "Food-irrelevant":
                    TN += 1
                else:
                    FN += 1
            else:
                if reviews['label'] == "Food-relevant":
                    TP += 1
                else:
                    FP += 1
    
    print 'Confusion Matrix'
    print TP , FP
    print FN , TN
    
    print 'Relevant Statistics: '
    print 'Accuracy= ', (TP+TN)/(float(TP+FP+FN+TN))
    print 'Relevant-Precesion= ' , TP / (float(TP+FP))
    print 'Relevant-Recall= ', TP/(float(TP+FN))
    
    print 
    
    print 'Irrelevant Statistics: '
    print 'Ir-Relevant-Precesion= ' , TN / (float(TN+FN))
    print 'Ir-Relevant-Recall= ', TN/(float(TN+FP))

print 'Started Testing...'
ScoringMethod = 2 # 1- Euclidean Distance Based,  2- Cosine Similiarity Based, 3- Manhattan Distance Based
TestRocchio('./testing_data.json',Mean_Classes_List, TrainingVocab, ScoringMethod)
print 'Ended Testing.'


# 
# **Euclidean Distance Based Scoring **
# 
# Relevant Statistics: 
# 
# Accuracy=  0.71
# 
# Relevant-Precesion=  0.809523809524
# 
# Relevant-Recall=  0.75
# 
# 
# Irrelevant Statistics: 
# 
# Ir-Relevant-Precesion=  0.540540540541
# 
# Ir-Relevant-Recall=  0.625
# 
# **Manhattan Distance Based Scoring**
# 
# Relevant Statistics: 
# 
# Accuracy=  0.72
# 
# Relevant-Precesion=  0.722222222222
# 
# Relevant-Recall=  0.955882352941
# 
# Irrelevant Statistics: 
# 
# Ir-Relevant-Precesion=  0.7
# 
# Ir-Relevant-Recall=  0.21875
# 
# 
# **Cosine Similairty Based Scoring**
# 
# Relevant Statistics: 
# 
# Accuracy=  0.66
# 
# Relevant-Precesion=  0.803571428571
# 
# Relevant-Recall=  0.661764705882
# 
# 
# Irrelevant Statistics: 
# 
# Ir-Relevant-Precesion=  0.477272727273
# 
# Ir-Relevant-Recall=  0.65625
# 

# # Part 3: Naive Bayes vs. Rocchio [20 points]
# 
# Which method gives the better results? In terms of what? How did you compare them? Can you explain why you observe what you do? Write 1-3 paragraphs below.

# **Add your answer here:**
# 
# The *Naive Bayes* method gives better results in terms of both the Accuracy of overall system and also the Precision and Recall of Relevant as well as Irrelvant Class Classification.
# 
# The comparision between the two classifiers was done based on the Accuracy first. The Naive Bayes based classifier resulted in being 90% accuracte to detect the class of the reviews of the testing dataset, whereas the Rocchio classifier got maximum of 72% accuracy when we used the Manhattan distance based method, where in case of Euclidean distance based method resulted in around 71% accuracy.
# 
#    Moreover, the comparision was also based on the Precision and Recall of both the Relevant Classification as well as the Irrelevant classification. The Naive bayes calssifier for Relevant classification has better precision as well as better recall.
# The Naive Bayes makes sure that the results are both precise as well as have good recall, thus leading to higher accuracy. Where as in case of the Rochiio, we have different Precision and Recall values for different scoring methods, but all of the precision and recall values are less than that of the Naive Bayes Classifier based classification.
# 
# The problem with the Manhattan Distance is the Recall of Irrelevant reviews is below than that of 0.5, which is very bad for a system. The Rochhio classifier based in euclidean based distance resulted in same accuracy but better recall and moderate precision.
# 
# The Naive Bayes Classifier is better than that of Rocchio based classifier because of various reasons:
# 
# 1) Naive Bayes is Text Classifier is probabilistic method based classifier, whereas the Rocchio classifier is based on the contiguity hypothesis. So the Rocchio Classifier will fail in case where there is no distinct distinguishing boundaries between different classes. Herein we have text based classification, which doesn't have the exact boundaries like many other classification problems. Since the words/tokens can appear in both the Relevant as well as Irrelevant reviews, we need a method which will use some probalistic model to predict their weightage, which is not in the case of Rocchio. That is why we are getting better results in terms of Accuracy, Precision, Recall in Niave Bayes as compared to Rocchio.
# 
# 2) The next reason of Rocchio giving bad results is that, classes in Rocchio Classification must be approximate spheres with similiar radii. Since the classification is done on the basis of the distance from the centroid of the class, then there are cases where on class is highly spread and other is not, but since the radius of highly spread class can be away from the test data point, hence resulting in the mis classification.
# 
# 3) One more problem of Rocchio is the multimodal class, if the dataset represents a class into two clusters, the resulting centroid we get will result is somewhere between the thwo clusters, thus misclassification happens.
# 
# 

# # Part 4: Recommenders [10 points]
# 
# Finally, since we've begun our discussion of recommenders, let's do a quick problem too:
# 
# The table below is a utility matrix, representing the ratings, on a 1–5 star scale, of eight items, *a* through *h*, by three users *A*, *B*, and *C*. 
# <pre>
# 
#   | a  b  c  d  e  f  g  h
# --|-----------------------
# A | 4  5     5  1     3  2
# B |    3  4  3  1  2  1
# C | 2     1  3     4  5  3
# 
# </pre>
# 
# Compute the following from the data of this matrix.
# 
# (a) Treating the utility matrix as boolean, compute the Jaccard distance between each pair of users.
# 
# (b) Repeat Part (a), but use the cosine distance.
# 
# (c) Treat ratings of 3, 4, and 5 as 1 and 1, 2, and blank as 0. Compute the Jaccard distance between each pair of users.
# 
# (d) Repeat Part (c), but use the cosine distance.
# 
# (e) Normalize the matrix by subtracting from each nonblank entry the average
# value for its user.
# 
# (f) Using the normalized matrix from Part (e), compute the cosine distance
# between each pair of users.
# 
# (g) Which of the approaches above seems most reasonable to you? Give a one or two sentence argument supporting your choice.
**Add your answer here:**

(a) J(A,B) = J(B,C) = J(A,C) = 1/2 = 0.5

(b) Cos(A,B) = Cos(A,C) = Cos(B,C) = 2/3 = 0.67

(c) J(A,B) = 2/5 =  0.4
    J(A,C) = 2/6 =  0.33
    J(B,C) = 1/6 =  0.167

(d) Cos(A,B) = 1/(sqrt(3))    = 0.577
    Cos(A,C) = 1/2 = 0.5      = 0.5
    Cos(B,C) = 1/(2 *sqrt(3)) = 0.2886

(e)   | a     b     c      d     e      f     g       h
    --|---------------------------------------------------
    A | 2/3   5/3   -     5/3   -7/3    -     -1/3   -4/3
      |
    B | -     2/3   5/3   2/3   -4/3   -1/3   -4/3    -
      |
    C | -1     -    -2     0      -     1       2      0
    

(f) Cos(A,B) = 13/(3 * sqrt(55))  = 0.5843
    Cos(A,C) = -1/(3 * sqrt(5))   = -0.1924
    Cos(B,C) = -19/(6 *sqrt(11))  = -0.954

(g) 