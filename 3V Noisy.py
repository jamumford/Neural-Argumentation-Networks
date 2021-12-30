# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 18:31:16 2020

@author: Truffles
"""

import numpy as np
import re
import time
import sys
import math
import random
import pickle

np.set_printoptions(threshold=sys.maxsize)

def loadframework(framework):
    # Read in the AF from a .tgf format file
    datafile = open('tgf/' + framework, 'r')
    lines = datafile.readlines()
    datafile.close()
    # Process the file to read the arguments and attacks into lists of strings 
    args = {}
    nisargs = []
    weights = []
    firstHalf = True
    depth = 0
    for lineidx, line in enumerate(lines):
        # Drop the '\n' from each line. Then everything before the '#' is an
        # argument, everything after is an attack.
        content = line[:(len(line)-1)]
        if content == '#':
            weights = np.zeros((depth, depth))
            firstHalf = False
        else:
            if firstHalf:
                args.update({lineidx : content})
                args.update({content : lineidx})
                nisargs.append(content)
                depth += 1
            else:
                attack = content.split()
                attacker = attack[0]
                attacked = attack[1]
                attackeridx = args[attacker]
                attackedidx = args[attacked]
                weights[attackeridx][attackedidx] = -1.0
    return (args, weights, depth, nisargs)

def loadextension(extension, argdict, numargs):
    extfile = open('solutions/' + extension, 'r')
    # extsets assumes that all extensions are contained in just one line
    extsets = extfile.readline()
    extfile.close()
    extensions = re.findall(r'\[.*?\]', extsets)
    labellings = []
    for exten in extensions:
        labelling = [0.5]*numargs
        inargs = re.findall(r'\w+', exten)
        for inarg in inargs:
            if inarg in argdict:
                labidx = argdict[inarg]
                labelling[labidx] = 1.0
        labellings.append(labelling)
    return labellings

def weightchecker(weightin, weightout, depth):
    truepos = 0
    falsepos = 0
    trueneg = 0
    falseneg = 0
    attacks = 0
    for bidx in range(depth):
        for aidx in range(depth):
            if weightin[aidx][bidx] == -1.0:
                if weightout[aidx][bidx] <= -0.5:
                    truepos += 1
                    attacks += 1
                else:
                    falseneg += 1
            else:
                if weightout[aidx][bidx] > -0.5:
                    trueneg += 1
                else:
                    falsepos += 1
                    attacks += 1
    #print("Attacks check: " + str(attacks))
    return truepos, trueneg, falsepos, falseneg, attacks

# Labellings used in Extensions examples section 3.2.
label1 = ([1, 0, 0.5, 0.5, 0.5])
label2 = ([1, 0, 0, 1, 0.5])
label3 = ([1, 0, 1, 0, 0])
labellings = []
labellings.append(label1)
labellings.append(label2)
labellings.append(label3)

# Weights are of the form such that each [a, b, c, ..., n] represents the attacked arguments.
weightvals = np.zeros((5, 5))

class NeuralNetwork:
    def __init__(self):
        self.weights = 0
        self.theta = -0.5
    
    
    def partlabel(self, labelseries, weights, proportion, setsize, corrupt):
        numeric = []
        if (proportion == 1.0):
            if setsize != "fixed":
                sys.exit("Cannot have variable labelling set sizes with 1.0 proportion!")
            for lab in labelseries:
                translation = self.ext2label(lab, weights)
                numeric.append(translation)
        else:
            depth = max(len(elem) for elem in labelseries)
            selection = math.ceil(depth * proportion)
            #print("Inital label size: " + str(depth))
            #print("Selection label size: " + str(selection))
            if setsize == "fixed":
                labcount = len(labelseries)
            else:
                labcount = setsize
            for i in range(labcount):        
                new = ["null"] * (depth - selection) + [0.5] * selection
                random.shuffle(new)
                old = []
                count = 0
                while ((new != old) and (count < 100)):
                    old = new.copy()
                    for bidx, bval in enumerate(old):
                        new[bidx] = self.confirm(bidx, old, weights)
                    count += 1
                numeric.append(new)
        if corrupt == 0.0:
            noisenumeric = numeric
        else:
            noisenumeric = []
            for labelling in numeric:
                noiselabelling = []
                possible = [1.0, 0.5, 0.0]
                for label in labelling:
                    if label == 1.0:
                        outcome = random.choices(possible, [1-corrupt, corrupt/2.0, corrupt/2.0])
                        noiselabelling.append(outcome[0])
                    elif label == 0.5:
                        outcome = random.choices(possible, [corrupt/2.0, 1-corrupt, corrupt/2.0])
                        noiselabelling.append(outcome[0])
                    elif label == 0.0:
                        outcome = random.choices(possible, [corrupt/2.0, corrupt/2.0, 1-corrupt])
                        noiselabelling.append(outcome[0])
                    else:
                        noiselabelling.append(label)
                noisenumeric.append(noiselabelling)
        return noisenumeric

    
    def ext2label(self, labelling, weights):
        labels = []
        for bidx, bval in enumerate(labelling):
            trial = bval - 1.0
            for aidx, aval in enumerate(labelling):
                if aval == 1.0:
                    temp = (aval * weights[aidx][bidx])
                    if temp < trial:
                        trial = temp
            labels.append(trial + 1.0)
        return labels
    
    
    # Need to actually return confirmation that input is the same as output.
    def confirm(self, bidx, labelling, weights):
        trial = 0.0
        if (labelling[bidx] != "null"):
            for aidx, aval in enumerate(labelling):
                if (aval != "null"):
                    if weights[aidx][bidx] <= self.theta:
                        rounded = -1.0
                    else:
                        rounded = 0.0
                    net = (aval * rounded)
                    if net < trial:
                        trial = net
            output = trial + 1.0
        else:
            output = "null"
        return output
    
    
    # The function to intialise and execute an algorithm that outputs a defeat relation
    # consistent with the input labellings, or reveals the impossibility of a solution.
    def letsbegin(self, algorithm, labellings, limit):
        if algorithm == 'onlinenaive':
            networkoutput = self.onlinenaive(labellings, limit)
        elif algorithm == 'onlinestrict':
            networkoutput = self.onlinestrict(labellings, limit)
        elif algorithm == 'offlinenaive':
            networkoutput = self.offlinenaive(labellings, limit)
        elif algorithm == 'offlinestrict':
            networkoutput = self.offlinestrict(labellings, limit)
        return networkoutput
    
    
    # onlinenaive examines each labelling in turn. For each labelling, for each
    # misclassified argument each plausible attack is incremented.
    # Once an attack reaches the threshold it is implemented in the AF.
    def onlinenaive(self, labelseries, limit):
        depth = max(len(elem) for elem in labelseries)
        self.weights = np.zeros((depth, depth))
        accept = 0
        errors = 1
        self.theta = -0.5
        #eta = abs(self.theta / len(labelseries))
        eta = 0.01
        maxiter = math.ceil(abs(self.theta) / eta) + 2
        iteration = 0
        begin = time.time()
        elapse = time.time()
        while ((elapse - begin < limit) and (errors > accept)):
            errors = 0
            for labelling in labelseries:
                for bidx, bval in enumerate(labelling):
                    output = self.confirm(bidx, labelling, self.weights)
                    if (bval != output) and (bval != "null"):
                        errors += 1
                        for aidx, aval in enumerate(labelling):
                            if (aval != "null"):
                                if (bval == 1.0 and output == 0.5 and aval == 0.5)\
                                or (bval in (1.0, 0.5) and output == 0.0 and aval == 1.0):
                                    self.weights[aidx][bidx] += eta
                                elif (bval == 0.5 and output == 1.0 and aval == 0.5)\
                                or (bval == 0.0 and output in (1.0, 0.5) and aval == 1.0):
                                    self.weights[aidx][bidx] += -eta
                                if (self.weights[aidx][bidx] < -1.0):
                                    self.weights[aidx][bidx] = -1.0
                                elif (self.weights[aidx][bidx] > 0.0):
                                    self.weights[aidx][bidx] = 0.0
            elapse = time.time()
            iteration += 1
            if iteration > maxiter:
                accept += len(labelseries)
        #print ("number of errors: " + str(errors))
        return (self.weights, errors)
    
    
    # onlinestrict examines each labelling in turn. For each labelling, for each
    # misclassified argument each plausible attack is incremented and attacks 
    # that would be inconsistent are decremented.
    # Once an attack reaches the threshold it is implemented in the AF.
    def onlinestrict(self, labelseries, limit):
        depth = max(len(elem) for elem in labelseries)
        self.weights = np.zeros((depth, depth))
        accept = 0
        errors = 1
        self.theta = -0.5
        #eta = abs(self.theta / len(labelseries))
        eta = 0.01
        maxiter = math.ceil(abs(self.theta) / eta) + 2
        iteration = 0
        begin = time.time()
        elapse = time.time()
        while ((elapse - begin < limit) and (errors > accept)):
            errors = 0
            for labelling in labelseries:
                for bidx, bval in enumerate(labelling):
                    output = self.confirm(bidx, labelling, self.weights)
                    if (bval != output):
                        errors += 1
                    if (bval != "null"):
                        for aidx, aval in enumerate(labelling):
                            if (aval != "null"):
                                if (bval == 1.0 and aval in (1.0, 0.5)) or (bval == 0.5 and aval == 1.0):
                                    self.weights[aidx][bidx] += eta
                                elif (bval == 0.5 and output == 1.0 and aval == 0.5)\
                                or (bval == 0.0 and output in (1.0, 0.5) and aval == 1.0):
                                    self.weights[aidx][bidx] += -eta
                                if (self.weights[aidx][bidx] < -1.0):
                                    self.weights[aidx][bidx] = -1.0
                                elif(self.weights[aidx][bidx] > 0.0):
                                    self.weights[aidx][bidx] = 0.0
            elapse = time.time()
            iteration += 1
            if iteration > maxiter:
                accept += len(labelseries)
        #print ("number of errors: " + str(errors))
        return (self.weights, errors)
    
    
    # offlinenaive only amends after the entire labelling data set has been examined. 
    # For each labelling, for each misclassified argument each plausible attack 
    # is incremented. Once an attack reaches the threshold it is implemented in 
    # the AF.
    def offlinenaive(self, labelseries, limit):
        depth = max(len(elem) for elem in labelseries)
        self.weights = np.zeros((depth, depth))
        accept = 0
        errors = 1
        self.theta = -0.5
        #eta = abs(self.theta / len(labelseries))
        eta = 0.01
        maxiter = math.ceil(abs(self.theta) / eta) + 2
        iteration = 0
        begin = time.time()
        elapse = time.time()
        while ((elapse - begin < limit) and (errors > accept)):
            errors = 0
            delta = np.zeros((depth, depth))
            for labelling in labelseries:
                for bidx, bval in enumerate(labelling):
                    output = self.confirm(bidx, labelling, self.weights)
                    if (bval != output) and (bval != "null"):
                        errors += 1
                        for aidx, aval in enumerate(labelling):
                            if (aval != "null"):
                                if (bval == 1.0 and output == 0.5 and aval == 0.5)\
                                or (bval in (1.0, 0.5) and output == 0.0 and aval == 1.0):
                                    delta[aidx][bidx] += eta
                                elif (bval == 0.5 and output == 1.0 and aval == 0.5)\
                                or (bval == 0.0 and output in (1.0, 0.5) and aval == 1.0):
                                    delta[aidx][bidx] += -eta
            for bidx in range(depth):
                for aidx in range(depth):
                    self.weights[aidx][bidx] += delta[aidx][bidx]
                    if (self.weights[aidx][bidx] <= -1.0):
                        self.weights[aidx][bidx] = -1.0
                    elif (self.weights[aidx][bidx] >= 0.0):
                        self.weights[aidx][bidx] = 0.0
            elapse = time.time()
            iteration += 1
            if iteration > maxiter:
                accept += len(labelseries)
        #print ("number of errors: " + str(errors))
        return (self.weights, errors)
    
    # offlinestrict only amends after the entire labelling data set has been examined. 
    # For each labelling, for each misclassified argument each plausible attack is 
    # incremented and attacks that would be inconsistent are decremented.
    # Once an attack reaches the threshold it is implemented in the AF.
    def offlinestrict(self, labelseries, limit):
        depth = max(len(elem) for elem in labelseries)
        self.weights = np.zeros((depth, depth))
        accept = 0
        errors = 1
        self.theta = -0.5
        #eta = abs(self.theta / len(labelseries))
        eta = 0.01
        maxiter = math.ceil(abs(self.theta) / eta) + 2
        iteration = 0
        begin = time.time()
        elapse = time.time()
        while ((elapse - begin < limit) and (errors > accept)):
            errors = 0
            delta = np.zeros((depth, depth))
            for labelling in labelseries:
                for bidx, bval in enumerate(labelling):
                    output = self.confirm(bidx, labelling, self.weights)
                    if (bval != output):
                        errors += 1
                    if (bval != "null"):
                        for aidx, aval in enumerate(labelling):
                            if (aval != "null"):
                                if (bval == 1.0 and aval in (1.0, 0.5)) or (bval == 0.5 and aval == 1.0):
                                    delta[aidx][bidx] += eta
                                elif (bval == 0.5 and output == 1.0 and aval == 0.5)\
                                or (bval == 0.0 and output in (1.0, 0.5) and aval == 1.0):
                                    delta[aidx][bidx] += -eta
            for bidx in range(depth):
                for aidx in range(depth):
                    self.weights[aidx][bidx] += delta[aidx][bidx]
                    if (self.weights[aidx][bidx] <= -1.0):
                        self.weights[aidx][bidx] = -1.0
                    elif (self.weights[aidx][bidx] >= 0.0):
                        self.weights[aidx][bidx] = 0.0
            elapse = time.time()
            iteration += 1
            if iteration > maxiter:
                accept += len(labelseries)
        #print ("number of errors: " + str(errors))
        return (self.weights, errors)


# setparameters takes a string input = ["test", "100_0.5", "fix_1.0", "fix_0.5"] and 
# outputs the parameters for the code execution of the current experimental run.
# Outputs are [algorithmlist, frameworklist, extype, fractionlist, corruptlist, setsize]
def setparameters(instruct):
    if instruct == "test":
        algorithmlist = ['onlinenaive', 'onlinestrict']
        frameworklist = ['traffic3.tgf', 'traffic9.tgf']
        fractionlist = [0.5]
        corruptlist = [0.0, 0.05, 0.1]
        setsize = "fixed"
    else:
        algorithmlist = ['onlinenaive', 'onlinestrict', 'offlinenaive', 'offlinestrict']
        frameworklist = ['traffic1.tgf', 'traffic2.tgf', 'traffic3.tgf', 'traffic4.tgf',
                 'traffic5.tgf', 'traffic6.tgf', 'traffic7.tgf', 'traffic8.tgf',
                 'traffic9.tgf', 'traffic10.tgf', 'traffic11.tgf', 'traffic12.tgf']
        corruptlist = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        if instruct == "100_0.5":
            setsize = 100
            fractionlist = [0.5]
        elif instruct == "fix_1.0":
            setsize = "fixed"
            fractionlist = [1.0]
        elif instruct == "fix_0.5":
            setsize = "fixed"
            fractionlist = [0.5]
    extype = '.EE-CO'
    return algorithmlist, frameworklist, extype, fractionlist, corruptlist, setsize


def setmetrics(iteration, algorithmlist):
    accdata = []
    defprdata = []
    defredata = []
    nonprdata = []
    nonredata = []
    deff1data = []
    nonf1data = []
    mccdata = []
    timedata = []
    xaxis = len(iteration)
    for item in range(xaxis):
        accdata.append([])
        defprdata.append([])
        defredata.append([])
        nonprdata.append([])
        nonredata.append([])
        deff1data.append([])
        nonf1data.append([])
        mccdata.append([])
        timedata.append([])
        for algidx, algorithm in enumerate(algorithmlist):
            accdata[item].append([])
            defprdata[item].append([])
            defredata[item].append([])
            nonprdata[item].append([])
            nonredata[item].append([])
            deff1data[item].append([])
            nonf1data[item].append([])
            mccdata[item].append([])
            timedata[item].append([])
    return accdata, defprdata, defredata, nonprdata, nonredata, deff1data, nonf1data, mccdata, timedata


# runexperiment takes two inputs where the data is the initialised data framework that will store the outputs and
# parameters are the experimental setup [algorithmlist, frameworklist, extype, fractionlist, corruptlist, setsize].
# The function returns the outputs for the nine metrics [accdata, defprdata, defredata, nonprdata, nonredata, 
# deff1data, nonf1data, mccdata, timedata].
def runexperiment(data, paras):
    noisyfixlabs = []
    algorithmlist = paras[0]
    frameworklist = paras[1]
    extype = paras[2]
    fractionlist = paras[3]
    corruptlist = paras[4]
    setsize = paras[5]
    for fidx, framework in enumerate(frameworklist):
        limit = 60.0
        trialframework = loadframework(framework)
        extensions = loadextension(framework[0:framework.find('.')] + extype, trialframework[0], trialframework[2])
        trialnetwork = NeuralNetwork()
        if setsize == "fixed":
            for fracidx, fraction in enumerate(fractionlist):
                # Generate the labellings data set. Note the partlabel argument that sets the number of labels in each labelling.
                if corruptlist == "fixed":
                    numericlabs = trialnetwork.partlabel(extensions, trialframework[1], fraction, setsize, 0.0)
                    trial = runalgs(data, trialnetwork, trialframework, numericlabs, algorithmlist, fracidx, limit)
                    data = trial
                else:
                    for coridx, cor in enumerate(corruptlist):
                        numericlabs = trialnetwork.partlabel(extensions, trialframework[1], fraction, setsize, cor)
                        trial = runalgs(data, trialnetwork, trialframework, numericlabs, algorithmlist, coridx, limit)
                        data = trial
                        if fraction == 1.0:
                            noisyfixlabs.append(numericlabs)
        else:
            fraction = fractionlist[0]
            if corruptlist == "fixed":
                for i in range(10):
                    sizes = (i + 1) * 10
                    # Generate the labellings data set. Note the partlabel argument that sets the number of labels in each labelling.
                    numericlabs = trialnetwork.partlabel(extensions, trialframework[1], fraction, sizes, 0.0)
                    trial = runalgs(data, trialnetwork, trialframework, numericlabs, algorithmlist, i, limit)
                    data = trial
            else:
                for coridx, cor in enumerate(corruptlist):
                    numericlabs = trialnetwork.partlabel(extensions, trialframework[1], fraction, setsize, cor)
                    trial = runalgs(data, trialnetwork, trialframework, numericlabs, algorithmlist, coridx, limit)
                    data = trial
        print("Done data set: " + str(fidx))
    print(len(data))
    print(len(data[0]))
    print(len(data[0][0]))
    print(len(data[0][0][0]))
    pathway = "3VNoisy/3VNoisy" + str(setsize) + "_" + str(fractionlist[0]) + "data"
    with open(pathway + ".txt", 'w') as ftext:
        for item in data:
            ftext.write("%s\n" % item)
    # Printed as data[metric][x-axis][algidx][fidx]
    pickle.dump(data, open(pathway +".p","wb"))
    if fraction == 1.0:
        pickle.dump(data, open(pathway +"niskylabs.p","wb"))
    return data


def runalgs(data, trialnetwork, trialframework, numericlabs, algorithmlist, iteration, limit):
    for algidx, algorithm in enumerate(algorithmlist):
        start = time.time()
        networkoutput = trialnetwork.letsbegin(algorithm, numericlabs, limit)
        end = time.time()
        con = weightchecker(trialframework[1], networkoutput[0], trialframework[2])
        tp = con[0]
        tn = con[1]
        fp = con[2]
        fn = con[3]
        acc = (tp + tn) / (tp + tn + fp + fn)
        if (tp + fp != 0):
            defpr = tp / (tp + fp)
        else:
            defpr = 0
        if (tp + fn != 0):
            defre = tp / (tp + fn)
        else:
            defre = 0
        if (tn + fn != 0):
            nonpr = tn / (tn + fn)
        else:
            nonpr = 0
        if (tn + fp != 0):
            nonre = tn / (tn + fp)
        else:
            nonre = 0
        if (defpr + defre != 0):
            deff1 = 2 * (defpr * defre) / (defpr + defre)
        else:
            deff1 = 0
        if (nonpr + nonre != 0):
            nonf1 = 2 * (nonpr * nonre) / (nonpr + nonre)
        else:
            nonf1 = 0
        if ((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn)) != 0.0:
            mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
        else:
            mcc = 0
        data[0][iteration][algidx].append(acc)
        data[1][iteration][algidx].append(defpr)
        data[2][iteration][algidx].append(defre)
        data[3][iteration][algidx].append(nonpr)
        data[4][iteration][algidx].append(nonre)
        data[5][iteration][algidx].append(deff1)
        data[6][iteration][algidx].append(nonf1)
        data[7][iteration][algidx].append(mcc)
        data[8][iteration][algidx].append(end-start)
    return data


# Input for paras = ["test", "100_0.5", "fix_1.0", "fix_0.5"]
# Output for paras = [algorithmlist, frameworklist, extype, fractionlist, corruptlist, setsize]   
"""
paras = setparameters("test")
initial = setmetrics(paras[4], paras[0])
experiment0 = runexperiment(initial, paras)
"""

paras = setparameters("100_0.5")
initial = setmetrics(paras[4], paras[0])
experiment1 = runexperiment(initial, paras)

paras = setparameters("fix_1.0")
initial = setmetrics(paras[4], paras[0])
experiment2 = runexperiment(initial, paras)

paras = setparameters("fix_0.5")
initial = setmetrics(paras[4], paras[0])
experiment3 = runexperiment(initial, paras)


