# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 13:58:52 2017

@author: Katie Bug
"""

###############################
# ProHealth REU Summer 2017
# Kate Sanders & Rob Long
# BayesNets2
###############################

import numpy as np
import random

def main():
    file = "26node.txt"
    fr = file_reader(file)
    # list of lists containing the node names parent names and probailities 
    names_parents_probs = fr.read_file()
    # turns the list of strings to bayes_node objects
    network = get_network(names_parents_probs)
    finished = False
    while not finished:
        print("Nodes in Network:")
        print(', '.join(names_parents_probs[0]))
        query = get_query(names_parents_probs[0], network)
        evidence = get_evidence(names_parents_probs[0], network)
        choice = int(input("Would you like to: \
                           \n1. Perform variable elimination\
                           \n2. Perform likelihood weighting\n"))
        if choice == 1:
            elim_order = get_elim_order(names_parents_probs[0], network)
            ve = variable_elimination(query, evidence, elim_order, network)
            value = ve.exact_inference()
        else:
            sample_num = int(input("Enter the number of samples:\n"))
            lw = likelihood_weighting(network)
            value = lw.calculate_weight(query, evidence, sample_num)
        print("\n")
        print(nice_looking_probability_rep(query, evidence))
        print(str(value))
        print("\n")

# converts node names into the node object
def str_to_node(str_node, node_names, network):
    index = node_names.index(str_node)
    return network[index]

# compacts the originaly obtained data into bayes_node objects         
def get_network(names_parents_probs):
    network = []
    for i in range(len(names_parents_probs[0])):
        node = bayes_node(names_parents_probs[0][i], [],
                          names_parents_probs[2][i], 
                          names_parents_probs[3][i])
        network.append(node)
    # turns the strings of the parent list into the correct bayes_nodes
    for i in range(len(names_parents_probs[1])):
        parent_list = []
        for parent in names_parents_probs[1][i]:
            parent_list.append(str_to_node(
                    parent, names_parents_probs[0], network))
        network[i].parents = parent_list
    return network

# returns dictionary for evidence: bool           
def get_evidence(node_names, network):
    evidence_num = int(input("Enter the number of evidence values:\n"))
    evidence = {}
    for n in range(evidence_num):
        ok = False
        while not ok:
            ev = input("Enter evidence and its value:\n")
            ev_split = ev.split()
            if len(ev_split) == 2:
                ok = True
        str_node = ev_split[0]
        node = str_to_node(str_node.upper(), node_names, network)
        if ev_split[1].upper() == "T" or ev_split[1].upper() == "TRUE":
            evidence[node] = True
        else:
            evidence[node] = False
    return evidence

# returns dictionary for query : bool
def get_query(node_names, network):
    query = {}
    qu = input("Enter the query and its value:\n")
    qu_split = qu.split()
    str_node = qu_split[0]
    node = str_to_node(str_node.upper(), node_names, network)
    if qu_split[1].upper() == "T" or qu_split[1].upper() == "TRUE":
        query[node] = True
    else:
        query[node] = False
    return query

# returns the order nodes should be eliminated in the network
def get_elim_order(node_names, network):
    elim_order = []
    elim = input("Enter the elimination order, seperating each\
                           node with a space:\n")
    elim= elim.upper().split()
    for e in elim:
         e_node = str_to_node(e, node_names, network)
         elim_order.append(e_node)
    return elim_order

# formats the input into a nice P(A | B, C) format
def nice_looking_probability_rep(query, evidence):
    key_list_q = query.keys()
    key_list_e = evidence.keys()
    statement = "P( "
    for q in key_list_q:
        statement += q.name + " = " + str(query[q])
    statement += " | "
    for e in key_list_e:
        statement += e.name + " = " + str(evidence[e]) + ", "
    statement += ")"
    return statement
 
# makes a truth table. alternates 2^index
def make_truth_table(parent_num):
    table = np.ones((parent_num, pow(2, parent_num)), dtype = bool)
    for n in range(len(table)):
        for i in range(len(table[n])):
            if i != 0 and i % pow(2, n) == 0:
                table[n, i] = not table[n, i - 1]
            elif i != 0 and i % pow(2, n) != 0:
                table[n, i] = table[n, i - 1]
    return table

# turns truth table into a list of tuples corresponding to the rows
def make_truth_tuples(parents_num):
    tuple_list = []
    truth = make_truth_table(parents_num)
    for i in range(len(truth[0])):
        pre_tuple = []
        for j in range(len(truth)):
            pre_tuple.append(truth[j][i])
        tuple_list.append(tuple(pre_tuple))
    return tuple_list

# This class performs approximate inference through likelihood weighting
# This is much faster and less labor-intensive on the user's part than
# variable elimination
class likelihood_weighting:
    def __init__(self, network):
        self.evidence = []
        self.network = network
        self.weights = {}
    
    # main funciton of this class. Performs sampling sample_num times & 
    # returns the approximate probabilty of the query
    def calculate_weight(self, query, evidence, sample_num):
        self.evidence = evidence
        evidence_info = self.get_evidence_information()
        for i in range(sample_num):
            self.add_to_weight_dict(evidence_info)
        prob = self.prob_event(list(query.keys())[0].name)
        if query[list(query.keys())[0]] == True:
            return prob
        return 1 - prob
    
    # turns the needed information about the evidence into a list so that
    # this process doesn't have to be repeated for every sample
    def get_evidence_information(self):
        evidence_info = []
        for n in self.network:
            if n in self.evidence:
                sub_info = [n]
                if n.probs_ind == None:
                    p_indexes = []
                    for p in n.parents:
                        p_index = self.network.index(p)
                        p_indexes.append(p_index)
                    sub_info.append(tuple(p_indexes))
                evidence_info.append(sub_info)
        return evidence_info
    
    # this is a single sample run    
    def add_to_weight_dict(self, evidence_info):
        bool_list = self.get_likelihood_bool_list()
        key_tuple = self.get_likelihood_key_tuple(bool_list)
        weight = self.get_likelihood_weights(bool_list, evidence_info)
        if key_tuple in self.weights:
            self.weights[key_tuple] += weight
        else:
            self.weights[key_tuple] = weight
            
    # returns a list of booleans for the sample run. The value of each node
    # not in evidence is determined randomly based on the node's probability.
    def get_likelihood_bool_list(self):
        bool_list = []
        none_index_list = []
        for n in range(len(self.network)):
            if self.network[n] in self.evidence:
                bool_list.append(self.evidence[self.network[n]])
            else:
                if self.network[n].probs_ind != None:
                    rand_num = random.random()
                    if rand_num > self.network[n].probs_ind:
                        bool_list.append(False)
                    else:
                        bool_list.append(True)
                else:
                    bool_list.append(None)
                    none_index_list.append(n)
        #print(bool_list)
        while len(none_index_list) > 0:
            for i in none_index_list:
                node_none = self.network[i]
                parents_bool = []
                for p in node_none.parents:
                    parents_bool.append(bool_list[self.network.index(p)])
                if None not in parents_bool: 
                    cond_probs = node_none.probs_cond[tuple(parents_bool)]
                    rand_num = random.random()
                    if rand_num > cond_probs:
                        bool_list[i] = False
                    else:
                        bool_list[i] = True
                    none_index_list.remove(i)
        return bool_list
      
    # turns the the boolean list into a tuple for easy dictionary lookup
    def get_likelihood_key_tuple(self, bool_list):
        key_list = []
        for i in range(len(bool_list)):
            if bool_list[i] == False:
                name = "~" + self.network[i].name
            else:
                name = self.network[i].name
            key_list.append(name)
        return tuple(key_list)

    # the likelihood weight is the product of the likelihoods of each 
    # evidence variable being its set state depending on the randomly
    # generated sample            
    def get_likelihood_weights(self, bool_list, evidence_information):
        weight = 1
        for e in evidence_information:
            value = 1
            # if len(e) == 2 the node has parents
            if len(e) == 2:
                bools = []
                # e[2] contains the tuple with all parent indexes
                for p in e[1]:
                    bools.append(bool_list[p])
                value = e[0].probs_cond[tuple(bools)]
            else:
                value = e[0].probs_ind
            if self.evidence[e[0]] == False:
                value = 1 - value
            weight *= value
        return weight

    # returns probability of an event happening given the dictionary of weights    
    def prob_event(self, event):
        event_total_weight = 0
        for key in self.weights:
            if event in key:
                event_total_weight += self.weights[key]
        return (event_total_weight / self.get_weight_sum())
      
    # returns sum of all of the weights calculated    
    def get_weight_sum(self):
        total = 0
        for w in self.weights:
            total += self.weights[w]
        return total
    
# an exact inference algorithm given the order of the nodes to be eliminated
class variable_elimination:
    def __init__(self, query, evidence, elim_order, network):
        # a dictionary of query : value
        self.query = query
        # a dictionary of evidence : value
        self.evidence = evidence
        # a list of the order that nodes should be eliminated
        self.elim_order = elim_order
        # a list of all nodes
        self.network = network
        # a list of tuples of all nodes needed to calculate the posterior &
        # the parents of the node
        self.joint = []
        # initializes the joint probability
    
    # the main funciton for performing exact inference. If you want to view the
    # tables of vaues after each stage of the process, uncomment the commented 
    # sections in this funciton        
    def exact_inference(self):
        is_simple = self.get_joint()
        if is_simple and len(self.evidence) > 0:
            return self.simple_cpt()
        while len(self.elim_order) > 0:
            elim = self.elim_order.pop(0)
            name = "f" + elim.name
            mult = self.get_nodes_to_multiply(elim)
            function = self.mult_tables(mult, name)
#            print("    MULTIPLIED")
#            for p in function.parents:
#                print(p.name)
#            for z in function.probs_cond:
#                print(z, function.probs_cond[z])
            function = self.summ_out(function, elim)
            self.joint.append((function, function.parents))
#            print("    SUMMED OUT")
#            print(name)
#            for p in function.parents:
#                print(p.name)
#            for z in function.probs_cond:
#                print(z, function.probs_cond[z])
        final_mult = []
        for j in self.joint:
            final_mult.append(j[0])
        final = self.mult_tables(final_mult, "final")
#        print("\n       FINAL")
#        for p in final.parents:
#            print(p.name)
#        for f in final.probs_cond:
#            print(f, final.probs_cond[f])
        return self.get_final_value(final)
            
    # determines whether the value can be calculated with just the query's CPT
    def is_simple(self, query_parents, evidence):
        for q in query_parents:
            if q not in evidence:
                return False
            evidence.remove(q)
        if len(evidence) > 0:
            return False
        return True
    
    # Gets the joint probability and also returns a boolean indicating
    # whether further calcuations need to be made.
    # The joint probability list has the query and evidence nodes as well
    # as the parents of each
    def get_joint(self):
        joint = []
        evi = []
        qu_par = []
        for q in self.query.keys():
            joint.append(q)
        for e in self.evidence.keys():
            joint.append(e)
            evi.append(e)
        for j in joint:
            for p in j.parents:
                qu_par.append(p)
                if p not in joint:
                    joint.append(p)
        for j in joint:
            self.joint.append((j, j.parents))
        return self.is_simple(qu_par, evi)

    # used when you just need to look up a CPT value
    def simple_cpt(self):
        query_items = list(self.query.keys())
        cpt = query_items[0].probs_cond
        cpt_parents = query_items[0].parents
        truth = []
        for p in cpt_parents:
            truth.append(self.evidence[p])
        truth = tuple(truth)
        if self.query[query_items[0]] == True:
            return cpt[truth]
        return 1 - cpt[truth]
    
    # returns all nodes that contain elim node in the CPT 
    def get_nodes_to_multiply(self, elim):
        mult = []
        remove = []
        for j in self.joint:
            if j[0].name == elim.name:
                mult.append(j[0])
                remove.append(j)
            else:
                for p in j[1]:
                    if p == elim and j[0] not in mult:
                        mult.append(j[0])
                        remove.append(j)
        for r in remove:
            self.joint.remove(r)
        return mult

    def mult_tables(self, mult, func_name):
        all_parents = self.get_all_parents(mult)
        # contains all nodes with cpt tuples ready to be multiplied
        sub_table_list = []
        # corresponding parent indexes of each sub_table_list node according
        # to the parent's place in all_parents
        sub_table_parent_indexes = []
        # mult contains all nodes to be multiplied
        for m in mult:
            # the strings here are arbitrary; they were useful for debugging.
            if m.name[0] != "f":
                name = "t" + m.name
                new_node = self.make_new_mult_node(m, name)
                sub_table_list.append(new_node)
            elif m not in sub_table_list:
                sub_table_list.append(m)
        for s in sub_table_list:
            sub_indexes = []
            for p in s.parents:
                sub_indexes.append(all_parents.index(p))
            sub_table_parent_indexes.append(sub_indexes)
        prob_dict = {}
        # the following code is very inefficient and should be revised.
        if len(all_parents) >= 1:
            # creates ordering for the new multiplied table
            tuple_list = make_truth_tuples(len(all_parents))
            for t in tuple_list:
                value = 1
                # looks at the cpt of each node that needs to be multiplied
                # and matches up the boolean tuples with the new table
                for i in range(len(sub_table_list)):
                    for pr in sub_table_list[i].probs_cond:
                        if self.is_mult_match(pr,
                                              sub_table_parent_indexes[i], t):
                            value *= sub_table_list[i].probs_cond[pr]
                prob_dict[t] = value
            function = bayes_node(func_name, all_parents, None, prob_dict)
            return function
        else:
            print(mult)
            
    # returns a list of all parent nodes in the table    
    def get_all_parents(self, mult):
        par = []
        for m in mult:
            if m not in par and m.name[0] != "f":
                par.append(m)
            for p in m.parents:
                if p not in par:
                    par.append(p)
        return par
        
    # takes a node and makes a new node with the probs_cond containing all 
    # truth tuples and T F values of original node. 
    # This makes multiplying easier
    def make_new_mult_node(self, node, name):
        cond_prob = {}
        parents = []
        if len(node.parents) == 0:
            parents.append(node)
            table = make_truth_tuples(1)
            cond_prob[table[0]] = node.probs_ind
            cond_prob[table[1]] = 1 - node.probs_ind
        else:
            for p in node.parents:
                parents.append(p)
            parents.append(node)
            table = make_truth_tuples(len(parents))
            n = 0
            while n < (len(table) / 2):
                spot = list(table[n])
                spot = spot[:-1]
                cond_prob[table[n]] = node.probs_cond[tuple(spot)]
                n += 1
            while n < len(table):
                spot = list(table[n])
                spot = spot[:-1]
                cond_prob[table[n]] = 1 - node.probs_cond[tuple(spot)]
                n += 1
        return bayes_node(name, parents, None, cond_prob)
    
    # determines whether the tuple values of the multiplied node are
    # consitent the with tuple values of a given row in the multiplied table
    def is_mult_match(self, truth_sub, sub_indexes, truth_tuple):
        n = 0
        while n < len(sub_indexes):
            if truth_sub[n] != truth_tuple[sub_indexes[n]]:
                return False
            n+=1
        return True

    # eliminates the given value from a multiplied table by summing it out
    def summ_out(self, function, elim):
        prob_dict = {}
        parent_probs = function.probs_cond
        tuple_list = make_truth_tuples(len(function.parents) - 1)
        wanted_indexes = self.get_wanted_indexes(function, elim)
        for t in tuple_list:
            value = 0
            for p in parent_probs:
                if self.is_match(t, p, wanted_indexes):
                    value += parent_probs[p]
            prob_dict[t] = value
        function.parents.remove(elim)
        function.probs_cond = prob_dict
        return function
                    
    def is_match(self, truth_tuple, parent_tuple, wanted_indexes):
        n = 0
        while n < len(wanted_indexes):
            if truth_tuple[n] != parent_tuple[wanted_indexes[n]]:
                return False
            n+=1
        return True
            
    def get_wanted_indexes(self, function, elim):
        indexes = []
        i = function.parents.index(elim)
        for n in range(len(function.parents)):
            if n != i:
                indexes.append(n)
        return indexes
    
    # looks up the final value of the query
    def get_final_value(self, final_node):
        for e in self.evidence:
            i = final_node.parents.index(e)
            remove_list = []
            for tu in final_node.probs_cond:
                if self.evidence[e] != tu[i]:
                    remove_list.append(tu)
            for r in remove_list:
                final_node.probs_cond.pop(r)
        for q in self.query:
            i = final_node.parents.index(q)
            true_false = [0 , 0]
            for qu in final_node.probs_cond:
                if qu[i] == True:
                    true_false[0] = (final_node.probs_cond[qu])
                elif qu[i] == False:
                    true_false[1] = (final_node.probs_cond[qu])  
            if self.query[q] == True:
                    return self.hyperparameter(true_false[0], true_false[1])
            return self.hyperparameter(true_false[1], true_false[0])
        
    # adjusts the final value
    def hyperparameter(self, final_1, final_2):
        return (final_1 / (final_1 + final_2))
    
# simple object that holds both nodes in the bayesian network and functions
# created from multiple nodes multiplied together
class bayes_node:
    def __init__(self, name, parents_list, probs_ind, probs_cond):
        #string
        self.name = name
        #list
        self.parents = parents_list
        #float if independent, else None
        self.probs_ind = probs_ind
        #dictionary mapping tuple truth combo to value
        self.probs_cond = probs_cond

# used to read the formated text files
class file_reader:
    def __init__(self, file):
        self.file = file
        self.big_dict = {}
    
    def get_lines(self):
        f = open(self.file)
        lines = f.readlines()
        return lines

    def read_file(self):
        lines = self.get_lines()
        names = []
        parents = []
        probs_ind = []
        probs_cond = []
        for line in lines:
            index = 0
            line = line.upper()
            words = line.split()
            for word in words:
                if word == "NODE":
                    name = words[index + 1]
                    names.append(name)
                elif word == "PARENTS":
                    sub_parents = []
                    i = index + 1
                    while words[i] != ";":
                        sub_parents.append(words[i])
                        i+=1
                    parents.append(sub_parents)
                elif word == "PROBS":
                    sub_probs = []
                    i = index + 1
                    while words[i] != "};":
                        sub_probs.append(float(words[i]))
                        i+=1
                    if len(sub_probs) == 1:
                        probs_ind.append(sub_probs[0])
                        probs_cond.append({})
                    else:
                        probs_cond.append(self.make_cond_probs(
                                sub_probs, len(sub_parents)))
                        probs_ind.append(None)
                index+=1
        return [names, parents, probs_ind, probs_cond]
    
    def make_cond_probs(self, sub_probs, parents_num):
        cond = {}
        tuple_list = make_truth_tuples(parents_num)
        for p in range(len(sub_probs)):
            cond[tuple_list[p]] = sub_probs[p]
        return cond
 
main()
