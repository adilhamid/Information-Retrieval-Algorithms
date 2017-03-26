# Here define your function for building the graph by parsing the input file of tweets

import json
from pprint import pprint

''' Input : filename of the tweets
    Output : graph and user_id and name/screen_name mapping    
'''
def parse_json_file(filename,userid_names_dict , userid_screennames_dict ):
    userid_adjancency_graph = {}
    for line in open(filename, 'r'): # Reading each tweet one by one
        tweets = json.loads(line)
        
        # Reading the userid of the tweet, and also checking the mentions of the tweet : Here adding only the screen_name
        userid_screennames_dict.setdefault(tweets['user']['id'], set()).add(tweets['user']['screen_name'])

        #Checking the Mentions of all the tweets and storing the user mentioned to make the graph and also storing the 
        # mentioned users name and screen_name.
        for mentions in tweets['entities']['user_mentions']:
            
            userid_names_dict.setdefault(mentions['id'], set()).add(mentions['name'])      # Storing the Actual Name of user
            
            userid_screennames_dict.setdefault(mentions['id'], set()).add(mentions['screen_name']) # Storing the Screen Name of user
            
            if(mentions['id'] != tweets['user']['id']):
            
                userid_adjancency_graph.setdefault(tweets['user']['id'],[set() for _ in xrange(2)])[0].add(mentions['id']) # making the adjanceny graph
                
                userid_adjancency_graph.setdefault(mentions['id'],[set() for _ in xrange(2)])[1].add(tweets['user']['id']) # [0] index of graph represents out links and [1] represents in-links
    return userid_adjancency_graph



# Now We already have the graph and the user_id and screen_name/name mapping
def get_graph_from_dict(filename):
    userid_names_dict = {}
    userid_screennames_dict = {}
    userid_adjancency_graph = parse_json_file(filename,userid_names_dict, userid_screennames_dict)
    print 'Number of Nodes: ' , len(userid_adjancency_graph)
    num_edges = 0
    for node in userid_adjancency_graph:
        num_edges += len(userid_adjancency_graph[node][0])
    print 'Number of Edges: ', num_edges
    return userid_adjancency_graph

graph = get_graph_from_dict('pagerank.json')



# Getting the graph using the networks library #### if we use network graph based pagerank##################
import networkx as net

# Generate the graph using the network library from the ditionary already generated

def generate_network(graph):
    num_nodes = len(graph)
    network_graph = net.DiGraph()
    for node in graph:

        outgoing_nodes = graph[node][0]

        for nei_nodes in outgoing_nodes:
                network_graph.add_edge(node , nei_nodes)
   
    print 'Number of Nodes: ', len(network_graph.nodes())
    print 'Number of Edges: ', len(network_graph.edges())
    return network_graph

net_graph = generate_network(graph)


############################################Self Defined GRAPH##############################################################

import copy
import operator

'''
Input -> graph in form of adjaceny List
d = damping factor
epsilon - to stopping criteria
k ==== How many results you want
stop_flag - 0 --> Using stopping criteria as using epsilon
            1 --> Using Top k Results based on scoring change for checking stopping
            2 --> usinng top k results based on id change.

Output: 10 ranking list

'''

def PageRanker(graph, d , epsilon, k, stop_flag):
    num_nodes = len(graph)
    initial_weight =  ((1-float(d)) /(float)(num_nodes))  # Precomputed Fixed Weight
    pagerank = {}
    
    initial_rank = (1/float (num_nodes))   # Precomputed Initial Rank
    top_k_items = {}
    
    for nodes in graph:
        pagerank[nodes] = initial_rank
    
    for cnt in range(500):     # Top Number of Iterations
        pagerank_prev = copy.deepcopy(pagerank)    # Deep Copy to store the old values
        
        for nodes in graph:
            sum_rank = 0;
            
            in_links = graph[nodes][1]    # Getting the inlinks of a node to calculate the rank
            
            for n in in_links:
                num_outgoing_links = len(graph[n][0])   # Since we need to distribute the weights among all outgoing links.
                if num_outgoing_links > 0:
                    sum_rank +=  (1/(float(num_outgoing_links))) * pagerank_prev[n]
                    
            pagerank[nodes] = initial_weight + d * sum_rank   # Ranking updation
        
        if stop_flag == 0:

            sum_error = 0
            for nodes in graph:
                sum_error += pow(pagerank[nodes] - pagerank_prev[nodes], 2)

            if sum_error < num_nodes * pow(epsilon, 2):        # Stopping Condition
                break
        elif stop_flag == 1:
            sum_error = 0
            top_k_items = sorted(pagerank.items(), key = operator.itemgetter(1), reverse = True)[:k]
            
            for nodes,vals in top_k_items:
                sum_error += vals - pagerank_prev[nodes]
            if sum_error ==0:
                break;
                
        elif stop_flag == 2:
            
            top_k_items = sorted(pagerank.items(), key = operator.itemgetter(1), reverse = True)[:k]
            top_k_prev_items = sorted(pagerank_prev.items(), key = operator.itemgetter(1), reverse = True)[:k] 
            flag_break = True
            for i in range (0,k):
                if top_k_items[i][0] != top_k_prev_items[i][0]:
                    flag_break = False
            if flag_break == True:
                break
                    
        else:
            print 'Wrong Choice of stopping Condition'
            
            

    # Rank Normalization 
    normalization_factor = 0
    for node in pagerank:
        normalization_factor = normalization_factor + pagerank[node]
    
    for node in pagerank:
        pagerank[node] = pagerank[node]/normalization_factor
    
    # sort the ranks
    if stop_flag == 0:
        sorted_ranks = sorted(pagerank.items(), key = operator.itemgetter(1), reverse = True)
        top_k_items = sorted_ranks[:k]
    
    print 'Results: '
    for node in top_k_items:
        print 'User-Id: ' , node[0] , '\t Score: ' , node[1]
    print 'Total Iteration Requred: ', cnt

# Now let's call your function on the graph you've built. Output the results.

# As we have already build the graph in the second cell. We will just use the same
stopping_flag = 0     # 0 - criteria 1, 1- criteria 2 , 2- criteria 3 from the above cell
k = 10
epsilon = 0.000001
d= 0.9
print 'Started Ranking.......'
PageRanker(graph, d ,epsilon , k ,stopping_flag)



#################################PAGE RANK BASED ON NETWORK GRAPHS############################################################
import operator
import copy
   
def PageRanker_Network(graph, d ,epsilon):
    num_nodes = len(graph.nodes())
    initial_weight =  ((1-float(d)) /(float)(num_nodes)) 
    pagerank = {}
    
    initial_rank = (1/float (num_nodes))
    
    for nodes in graph.nodes():
        pagerank[nodes] = initial_rank
    
    for cnt in range(500):     # Top number of iterations
        pagerank_prev = copy.deepcopy(pagerank)
        for nodes in graph.nodes():
            sum_rank = 0;
            
            in_links = graph.in_edges(nodes)
            
            for n in in_links:
                num_incom_links = len(graph.out_edges(n[0]))
                if num_incom_links > 0:
                    sum_rank +=  (1/(float(num_incom_links))) * pagerank_prev[n[0]]
           
            pagerank[nodes] = initial_weight + d * sum_rank   # Ranking updation
        sum_error = 0
        for nodes in graph:
            sum_error += pow(pagerank[nodes] - pagerank_prev[nodes], 2)
         
        if sum_error < num_nodes * pow(epsilon, 2):        # Stopping Condition
            break
            

    # Rank Normalization 
    normalization_factor = 0
    for node in pagerank:
        normalization_factor = normalization_factor + pagerank[node]
    
    for node in pagerank:
        pagerank[node] = pagerank[node]/normalization_factor
    
    # sort the ranks     
    sorted_ranks = sorted(pagerank.items(), key = operator.itemgetter(1), reverse = True)
    sorted_ranks = sorted_ranks[:10]
    
    print 'Results: '
    for node in sorted_ranks:
        print 'User-Id: ' , node[0] , '\t Score: ' , node[1]
    print 'Total Iteration Requred: ', cnt

    
print 'Started Ranking.......'
PageRanker_Network(net_graph,0.9, 0.000001)
