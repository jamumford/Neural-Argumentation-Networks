# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 18:42:16 2018

Need to write code to parse the input labelling data into the numerical labels
used by class NeuralNetwork.
Then need to write code to translate output numerical labels to standard labels
and ensure they are associated with relevant argument.

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
                labelling[labidx] = 1
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
                #print("Weight is: " +str(weightout[aidx][bidx]))
                if weightout[aidx][bidx] > -0.5:
                    trueneg += 1
                else:
                    falsepos += 1
                    attacks += 1
    return truepos, trueneg, falsepos, falseneg, attacks

# Labellings used in Extensions examples section 3.2.
label1 = ([1.0, 0.5, 0.5, 0.5, 0.5])
label2 = ([1.0, 0.5, 0.5, 1.0, 0.5])
label3 = ([1.0, 0.5, 1.0, 0.5, 0.5])
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
        self.caminada = []
    
    def partlabel(self, labelseries, weights, proportion, setsize, corrupt):
        numeric = []
        if (proportion == 1.0):
            if setsize != "fixed":
                sys.exit("Cannot have variable labelling set sizes with 1.0 proportion!")
            for labelling in labelseries:
                numeric.append(labelling)
        else:
            depth = max(len(elem) for elem in labelseries)
            selection = math.ceil(depth * proportion)
            #print("Initial label size: " + str(depth))
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
                while ((new != old) and (count < 1000)):
                    old = new.copy()
                    new = self.confirm(old, weights)[1]
                    count += 1
                numeric.append(new)
        if corrupt == 0.0:
            noisenumeric = numeric
        else:
            noisenumeric = []
            for labelling in numeric:
                noiselabelling = []
                possible = [1.0, 0.5]
                for label in labelling:
                    if label == 1.0:
                        outcome = random.choices(possible, [1-corrupt, corrupt])
                        noiselabelling.append(outcome[0])
                    elif label == 0.5:
                        outcome = random.choices(possible, [corrupt, 1-corrupt])
                        noiselabelling.append(outcome[0])
                    else:
                        noiselabelling.append(label)
                noisenumeric.append(noiselabelling)
        return noisenumeric
    
    # Need to actually return confirmation that input is the same as output.
    def confirm(self, labelling, weights):
        hidden = self.feedforward(labelling, weights)
        for bidx, bval in enumerate(labelling):
            if labelling[bidx] == 1.0:
                hidden[bidx] = 1.0
        outputs = self.feedforward(hidden, weights)
        for bidx, bval in enumerate(labelling):
            if outputs[bidx] != "null":
                if outputs[bidx] < 1.0:
                    outputs[bidx] = 0.5
        return (hidden, outputs)
    
    # Performs the actual feedforward calculations from the neural network.
    def feedforward(self, inputs, weights):
        outputs = []
        for bidx, bval in enumerate(inputs):
            if bval == "null":
                outputs.append("null")
            else:
                trial = 0.0
                for aidx, aval in enumerate(inputs):
                    if aval != "null":
                        if weights[aidx][bidx] <= self.theta:
                            rounded = -1.0
                        else:
                            rounded = 0.0
                        net = (aval * rounded)
                        if net < trial:
                            trial = net
                outputs.append(trial + 1.0)            
        return outputs
    
    
    # The function to intialise and execute an algorithm that outputs a defeat relation
    # consistent with the input labellings, or reveals the impossibility of a solution.
    def letsbegin(self, algorithm, labellings, limit):
        if algorithm == 'onlinenoise2':
            networkoutput = self.onlinenoise2(labellings, limit)
        elif algorithm == 'offlinenoise2':
            networkoutput = self.offlinenoise2(labellings, limit)
        return networkoutput
    
    
    # onlineinc2 examines each labelling in turn. For each labelling, for each
    # misclassified argument each plausible attack is incremented.
    # Once an attack reaches the threshold it is implemented in the AF.
    def onlinenoise2(self, labelseries, limit):
        depth = max(len(elem) for elem in labelseries)
        self.weights = np.ones((depth, depth)) / -depth
        accept = 0
        begin = time.time()
        elapse = begin
        errors = 1
        delta = np.zeros((depth, depth))
        eta = 0.1
        while (elapse - begin < limit) and (errors > accept):
            errors = 0
            for labelling in labelseries:
                nodevals = self.confirm(labelling, self.weights)
                if nodevals[1] != labelling:
                    for bidx in range(depth):
                        for aidx in range(depth):
                            delta[aidx][bidx] = 0.0
                    for cidx, cval in enumerate(labelling):
                        if cval != nodevals[1][cidx]:
                            errors += 1
                            for bidx, bval in enumerate(labelling):
                                if (cval == 1.0 and bval == 1.0):
                                    delta[bidx][cidx] = eta / 2.0
                                elif cval != "null" and bval != "null" and nodevals[0][bidx] != 0.0:
                                    mu = eta * (cval - nodevals[1][cidx])
                                    delta[bidx][cidx] = mu                                   
                    for bidx, bval in enumerate(labelling):
                        if bval != "null":
                            if nodevals[0][bidx] != 1.0:
                                netb = 0.0
                                for cidx, cval in enumerate(labelling):
                                    if cval != "null":
                                        if (nodevals[0][bidx] == 0.5) or ((cval - nodevals[1][cidx]) < 0.0):
                                            netb += (cval - nodevals[1][cidx]) * self.weights[bidx][cidx]
                                for aidx, aval in enumerate(labelling):
                                    if aval == 1.0:
                                        mu = eta * netb
                                        delta[aidx][bidx] += mu
                            for aidx, aval in enumerate(labelling):
                                oldweight = self.weights[aidx][bidx].copy()
                                temp = oldweight + delta[aidx][bidx]
                                if (temp <= -1.0):
                                    self.weights[aidx][bidx] = -1.0
                                elif (temp >= 0.0):
                                    self.weights[aidx][bidx] = -0.01
                                else:
                                    self.weights[aidx][bidx] = temp
            elapse = time.time()
        #print ("number of errors: " + str(errors))
        #print ("prop of errors: " + str(errors/(depth*len(labelseries))))
        return (self.weights, errors)
    
    
    # onlineinc2 examines each labelling in turn. For each labelling, for each
    # misclassified argument each plausible attack is incremented.
    # Once an attack reaches the threshold it is implemented in the AF.
    def offlinenoise2(self, labelseries, limit):
        depth = max(len(elem) for elem in labelseries)
        self.weights = np.ones((depth, depth)) / -depth
        accept = 0
        begin = time.time()
        elapse = begin
        errors = 1
        eta = 0.1
        while (elapse - begin < limit) and (errors > accept):
            delta = np.zeros((depth, depth))
            errors = 0
            for labelling in labelseries:
                nodevals = self.confirm(labelling, self.weights)
                if nodevals[1] != labelling:
                    for cidx, cval in enumerate(labelling):
                        if (cval != nodevals[1][cidx]):
                            errors += 1
                            for bidx, bval in enumerate(labelling):
                                if (cval == 1.0 and bval == 1.0):
                                    delta[bidx][cidx] += eta / 2.0
                                elif cval != "null" and bval != "null" and (nodevals[0][bidx] != 0.0):
                                    mu = eta * (cval - nodevals[1][cidx])
                                    delta[bidx][cidx] += mu
                    for bidx, bval in enumerate(labelling):
                        if bval != "null":
                            if nodevals[0][bidx] != 1.0:
                                netb = 0.0
                                for cidx, cval in enumerate(labelling):
                                    if cval != "null":
                                        if (nodevals[0][bidx] == 0.5) or ((cval - nodevals[1][cidx]) < 0.0):
                                            netb += (cval - nodevals[1][cidx]) * self.weights[bidx][cidx]
                                for aidx, aval in enumerate(labelling):
                                    if aval == 1.0:
                                        mu = eta * netb
                                        delta[aidx][bidx] += mu
            for bidx, bval in enumerate(labelling):
                if bval != "null":
                    for aidx, aval in enumerate(labelling):
                        if aval != "null":
                            oldweight = self.weights[aidx][bidx].copy()
                            temp = oldweight + delta[aidx][bidx]
                            if (temp <= -1.0):
                                self.weights[aidx][bidx] = -1.0
                            elif (temp >= 0.0):
                                self.weights[aidx][bidx] = -0.01
                            else:
                                self.weights[aidx][bidx] = temp
            elapse = time.time()
        #print ("number of errors: " + str(errors))
        #print ("prop of errors: " + str(errors/(depth*len(labelseries))))
        return (self.weights, errors)
    


# setparameters takes a string input = ["test", "100_0.5", "fix_1.0", "fix_0.5"] and 
# outputs the parameters for the code execution of the current experimental run.
# Outputs are [algorithmlist, frameworklist, extype, fractionlist, corruptlist, setsize]
def setparameters(instruct):
    if instruct == "test":
        algorithmlist = ['onlinenoise2', 'offlinenoise2']
        frameworklist = ['traffic3.tgf', 'traffic9.tgf']
        fractionlist = [0.5]
        corruptlist = [0.0, 0.05, 0.1]
        setsize = "fixed"
    else:
        algorithmlist = ['onlinenoise2', 'offlinenoise2']
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
    pathway = "2VNoisy/2VNoisy" + str(setsize) + "_" + str(fractionlist[0]) + "data"
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


# Input for paras = ["test", "100_0.5", "fix_1.0"]
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
