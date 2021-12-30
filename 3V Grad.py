# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:21:08 2020

@author: Truffles
"""

import numpy as np
import re
import time
import sys
import random
import math
import pickle

np.set_printoptions(threshold=sys.maxsize)

def loadframework(framework):
    # Read in the AF from a .tgf format file
    datafile = open('tgf/' + framework, 'r')
    lines = datafile.readlines()
    datafile.close()
    # Process the file to read the arguments and attacks into lists of strings 
    args = {}
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
                depth += 1
            else:
                attack = content.split()
                attacker = attack[0]
                attacked = attack[1]
                attackeridx = args[attacker]
                attackedidx = args[attacked]
                weights[attackeridx][attackedidx] = -1
    return (args, weights, depth)

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
                if weightout[aidx][bidx] > -0.5:
                    trueneg += 1
                else:
                    falsepos += 1
                    attacks += 1
    return truepos, trueneg, falsepos, falseneg

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
    
    def partlabel(self, labelseries, weights, proportion, setsize):
        numeric = []
        if proportion == 1.0:
            if setsize != "fixed":
                sys.exit("Cannot have variable labelling set sizes with 1.0 proportion!")            
            for lab in labelseries:
                translation = self.ext2label(lab, weights)
                numeric.append(translation)
        else:
            depth = max(len(elem) for elem in labelseries)
            selection = math.ceil(depth * proportion)
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
                    for bidx, bval in enumerate(old):
                        new[bidx] = self.confirm(bidx, old, weights)
                    count += 1
                numeric.append(new)
        return numeric

    
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
    def letsbegin(self, algorithm, labellings):
        if algorithm == 'onlinestep':
            networkoutput = self.onlinestep(labellings)
        elif algorithm == 'onlineinc':
            networkoutput = self.onlineinc(labellings)
        elif algorithm == 'offline':
            networkoutput = self.offline(labellings)
        elif algorithm == 'greedy':
            networkoutput = self.greedy(labellings)
        return networkoutput
    
    # onlinestep examines each labelling in turn. For each labelling, for each
    # misclassified argument one attack is implemented such that the misclassified
    # argument is corrected and no previous labels of the argument are rendered
    # inconsistent. Currently the implemented attack is the first that is found to 
    # correct the label. Any attack found to be inconsistent with any previous
    # labelling is permanently fixed to zero.
    def onlinestep(self, labelseries):
        depth = max(len(elem) for elem in labelseries)
        self.weights = np.zeros((depth, depth))
        iteration = 0
        errors = 1
        wdelta = np.zeros((depth, depth))
        while ((iteration < 3) and (errors > 0)):
            errors = 0
            for labelling in labelseries:
                for bidx, bval in enumerate(labelling):
                    candidates = []
                    output = self.confirm(bidx, labelling, self.weights)
                    if (bval != output):
                        errors += 1
                    if (bval != "null"):
                        for aidx, aval in enumerate(labelling):
                            if (aval != "null"):
                                if wdelta[aidx][bidx] != 99.9:
                                    if (bval == 1.0 and aval in (1.0, 0.5)) or (bval == 0.5 and aval == 1.0):
                                        wdelta[aidx][bidx] = 99.9
                                        self.weights[aidx][bidx] = 0.0
                                    elif (bval == 0.5 and output == 1.0 and aval == 0.5)\
                                    or (bval == 0.0 and output in (1.0, 0.5) and aval == 1.0):
                                        candidates.append((aidx, bidx))
                        if (len(candidates) > 0):
                            elect = random.choice(candidates)
                            self.weights[elect[0]][elect[1]] = -1.0
            iteration += 1
        if (errors > 0):
            print ("NO R CAN BE CONSISTENT WITH THE DATA")
            print ("number of errors: " + str(errors))
        return (self.weights)
    
    
    # onlineinc examines each labelling in turn. For each labelling, for each
    # misclassified argument any attack capable of correcting the label if fully
    # implemented, is incrementally increased in magnitude so long as no previous 
    # labels of the argument are rendered inconsistent. Any attack found to be 
    # inconsistent with any previous labelling is permanently fixed to zero.
    def onlineinc(self, labelseries):
        depth = max(len(elem) for elem in labelseries)
        self.weights = np.zeros((depth, depth))
        self.theta = -0.5
        #eta = abs(self.theta / len(labelseries))
        eta = 0.01
        maxiter = math.ceil(abs(self.theta) / eta) + 2
        iteration = 0
        errors = 1
        wdelta = np.zeros((depth, depth))
        while ((iteration < maxiter) and (errors > 0)):
            errors = 0
            for labelling in labelseries:
                for bidx, bval in enumerate(labelling):
                    output = self.confirm(bidx, labelling, self.weights)
                    if (bval != output):
                        errors += 1
                    if (bval != "null"):
                        for aidx, aval in enumerate(labelling):
                            if (aval != "null"):
                                if (wdelta[aidx][bidx] != 99.9):
                                    if (bval == 1.0 and aval in (1.0, 0.5)) or (bval == 0.5 and aval == 1.0):
                                        wdelta[aidx][bidx] = 99.9
                                        self.weights[aidx][bidx] = 0.0
                                    elif (bval == 0.5 and output == 1.0 and aval == 0.5)\
                                    or (bval == 0.0 and output in (1.0, 0.5) and aval == 1.0):
                                        self.weights[aidx][bidx] += -eta
                                        if (self.weights[aidx][bidx] < -1.0):
                                            self.weights[aidx][bidx] = -1.0
            iteration += 1
        if (errors > 0):
            print ("NO R CAN BE CONSISTENT WITH THE DATA")
            print ("number of errors: " + str(errors))
        return (self.weights)

    
    # offline only amends after the entire labelling data set has been examined. 
    # For each possible attack a count is created that records the number of
    # misclassifications that would be corrected by the attack being implemented. 
    # If an attack would cause inconsistency with any other labelling then the count
    # is fixed at zero. After all labellings have been examined, for each argument 
    # all attacks are implemented that match the highest count of corrections.
    def offline(self, labelseries):
        depth = max(len(elem) for elem in labelseries)
        self.weights = np.zeros((depth, depth))
        wdelta = np.zeros((depth, depth))
        errors = 1
        iteration = 0
        maxiter = int((-1 + math.sqrt(1 + 8*depth))/2.0) + 2
        while ((iteration < maxiter) and (errors > 0)):
            refresh = np.where(wdelta != 99.9, 0, wdelta)
            wdelta = refresh
            errors = 0
            for labelling in labelseries:
                for bidx, bval in enumerate(labelling):
                    output = self.confirm(bidx, labelling, self.weights)
                    if (bval != output):
                        errors += 1
                    if (bval != "null"):
                        for aidx, aval in enumerate(labelling):
                            if (aval != "null"):
                                if (wdelta[aidx][bidx] != 99.9):
                                    if (bval == 1.0 and aval in (1.0, 0.5)) or (bval == 0.5 and aval == 1.0):
                                        wdelta[aidx][bidx] = 99.9
                                        self.weights[aidx][bidx] = 0.0
                                    elif (bval == 0.5 and output == 1.0 and aval == 0.5)\
                                    or (bval == 0.0 and output in (1.0, 0.5) and aval == 1.0):
                                        wdelta[aidx][bidx] += -1.0
            for bidx in range(depth):
                lowval = np.min(wdelta[:, bidx])
                if lowval < 0.0:
                    for aidx in range(depth):
                        if wdelta[aidx][bidx] == lowval:
                            self.weights[aidx][bidx] = -1.0
            iteration += 1
        if (errors > 0):
            print ("NO R CAN BE CONSISTENT WITH THE DATA")
        #print ("number of errors: " + str(errors))
        return self.weights
    
    # greedy only amends after the entire labelling data set has been examined. 
    # For each possible attack a count is created that records the number of
    # misclassifications would be corrected by the attack being implemented. If
    # an attack would cause inconsistency with any other labelling then the count
    # is fixed at zero. After all labellings have been examined, for each argument 
    # one attack is implemented that matches the highest count of corrections.
    # Currently this attack is the first to be matched.
    def greedy(self, labelseries):
        depth = max(len(elem) for elem in labelseries)
        self.weights = np.zeros((depth, depth))
        wdelta = np.zeros((depth, depth))
        errors = 1
        iteration = 0
        while ((iteration < (depth + 1)) and (errors > 0)):
            refresh = np.where(wdelta != 99.9, 0, wdelta)
            wdelta = refresh
            errors = 0
            for labelling in labelseries:
                for bidx, bval in enumerate(labelling):
                    output = self.confirm(bidx, labelling, self.weights)
                    if (bval != output):
                        errors += 1
                    if (bval != "null"):
                        for aidx, aval in enumerate(labelling):
                            if (aval != "null"):
                                if (wdelta[aidx][bidx] != 99.9):
                                    if (bval == 1.0 and aval in (1.0, 0.5)) or (bval == 0.5 and aval == 1.0):
                                        wdelta[aidx][bidx] = 99.9
                                        self.weights[aidx][bidx] = 0.0
                                    elif (bval == 0.5 and output == 1.0 and aval == 0.5)\
                                    or (bval == 0.0 and output in (1.0, 0.5) and aval == 1.0):
                                        wdelta[aidx][bidx] += -1.0
            for bidx in range(depth):
                candidates = []
                lowval = np.min(wdelta[:, bidx])
                if lowval < 0.0:
                    for aidx in range(depth):
                        if wdelta[aidx][bidx] == lowval:
                            candidates.append((aidx, bidx))
                if (len(candidates) > 0):
                    elect = random.choice(candidates)
                    self.weights[elect[0]][elect[1]] = -1.0
            iteration += 1
        if (errors > 0):
            print ("NO R CAN BE CONSISTENT WITH THE DATA")
        #print ("number of errors: " + str(errors))
        return self.weights
   

def runalgs(numericlabs, algorithmlist, iteration):
    for algidx, algorithm in enumerate(algorithmlist):
        start = time.time()
        networkoutput = trialnetwork.letsbegin(algorithm, numericlabs)
        end = time.time()
        con = weightchecker(trialframework[1], networkoutput, trialframework[2])
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
        accdata[iteration][algidx].append(acc)
        defprdata[iteration][algidx].append(defpr)
        defredata[iteration][algidx].append(defre)
        nonprdata[iteration][algidx].append(nonpr)
        nonredata[iteration][algidx].append(nonre)
        deff1data[iteration][algidx].append(deff1)
        nonf1data[iteration][algidx].append(nonf1)
        mccdata[iteration][algidx].append(mcc)
        timedata[iteration][algidx].append(end-start)
    return



algorithmlist = ['onlinestep', 'onlineinc', 'offline', 'greedy']
#algorithmlist = ['offline']


frameworklist = ['traffic1.tgf', 'traffic2.tgf', 'traffic3.tgf', 'traffic4.tgf',
                 'traffic5.tgf', 'traffic6.tgf', 'traffic7.tgf', 'traffic8.tgf',
                 'traffic9.tgf', 'traffic10.tgf', 'traffic11.tgf', 'traffic12.tgf']

#frameworklist = ['traffic10.tgf']
#extensionlist = ['.SE-GR', '.EE-PR', '.EE-CO']
extype = '.EE-CO'

fractionlist = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
fractionlist = [0.5]

setsize = "fixed"
setsize = 100

accdata = []
defprdata = []
defredata = []
nonprdata = []
nonredata = []
deff1data = []
nonf1data = []
mccdata = []
timedata = []

if setsize == "fixed":
    xaxis = len(fractionlist)
else:
    if len(fractionlist) > 1:
        sys.exit("Cannot have variable labelling set sizes with more than one proportion!") 
    xaxis = int(setsize/10)
    
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
           
for fidx, framework in enumerate(frameworklist):
    trialframework = loadframework(framework)
    extensions = loadextension(framework[0:framework.find('.')] + extype, trialframework[0], trialframework[2])
    T = framework[0:framework.find('.')] + extype
    trialnetwork = NeuralNetwork()
    if setsize == "fixed":
        for fracidx, fraction in enumerate(fractionlist):
            # Generate the labellings data set. Note the partlabel argument that sets the number of labels in each labelling.
            numericlabs = trialnetwork.partlabel(extensions, trialframework[1], fraction, setsize)
            runalgs(numericlabs, algorithmlist, fracidx)
    else:
        fraction = fractionlist[0]
        for i in range(10):
            sizes = (i + 1) * 10
            # Generate the labellings data set. Note the partlabel argument that sets the number of labels in each labelling.
            numericlabs = trialnetwork.partlabel(extensions, trialframework[1], fraction, sizes)
            runalgs(numericlabs, algorithmlist, i)
    print("Done data set: " + str(fidx))
 
    
# Printed as fulldata[metric][x-axis][algidx][fidx]
fulldata = [accdata, defprdata, defredata, nonprdata, nonredata, deff1data, nonf1data, mccdata, timedata]
print(len(fulldata))
print(len(fulldata[0]))
print(len(fulldata[0][0]))
print(len(fulldata[0][0][0]))


pathway = "3VGrad/3VGrad" + str(setsize) + "data"
with open(pathway + ".txt", 'w') as ftext:
    for item in fulldata:
        ftext.write("%s\n" % item)


pickle.dump(fulldata, open(pathway +".p","wb"))


"""
data = []            
for fidx, framework in enumerate(frameworklist):
    print(framework)
    data.append([framework]) 
    trialframework = loadframework(framework)
    extensions = loadextension(framework[0:framework.find('.')] + extype, trialframework[0], trialframework[2])
    T = framework[0:framework.find('.')] + extype
    trialnetwork = NeuralNetwork()
    for fracidx, fraction in enumerate(fractionlist):
        #print("prop is: " + str(fraction))
        data[fidx].append([fraction])
        numericlabs = trialnetwork.partlabel(extensions, trialframework[1], fraction)
        data[fidx][fracidx + 1].append(["Total labellings: " + str(len(numericlabs))])
        for algidx, algorithm in enumerate(algorithmlist):
            #print(algorithm)
            data[fidx][fracidx + 1].append([algorithm])
            if (algorithm == 'onlinestep'):
                start = time.time()
                networkoutput = trialnetwork.onlinestep(numericlabs)
                end = time.time()
            elif (algorithm == 'onlineinc'):
                start = time.time()
                networkoutput = trialnetwork.onlineinc(numericlabs)
                end = time.time()
            elif (algorithm == 'offline'):
                start = time.time()
                networkoutput = trialnetwork.offline(numericlabs)
                end = time.time()
            else:
                start = time.time()
                networkoutput = trialnetwork.greedy(numericlabs)
                end = time.time()
            z1 = weightchecker(trialframework[1], networkoutput, trialframework[2])
            data[fidx][fracidx + 1][algidx + 2].append([z1[2]])
            data[fidx][fracidx + 1][algidx + 2].append([z1[3]])
            data[fidx][fracidx + 1][algidx + 2].append([z1[4]])
            data[fidx][fracidx + 1][algidx + 2].append([z1[5]])
            data[fidx][fracidx + 1][algidx + 2].append([end - start])
    data[fidx].append(["Total args: " + str(trialframework[2])])
    data[fidx].append(["Total complete extensions: " + str(len(extensions))])
    data[fidx].append(["Total weights: ", (trialframework[2]**2)])

#print(data)
pickle.dump(data, open("3VGrad_0.5_1000lab.p","wb"))
""" 