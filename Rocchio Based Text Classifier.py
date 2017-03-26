
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