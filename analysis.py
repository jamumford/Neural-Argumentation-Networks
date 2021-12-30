# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 21:31:57 2021

@author: Truffles
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
import operator
import statistics
from scipy.stats import mannwhitneyu


# upload accepts a description of the experiments as input and returns the essential data
# that is required to perform the analysis.
# vartype = ["Labelling proportion", "Labelling size", 
# "Noise fixed 1.0", "Noise 100 0.5"]
# algtype = ["3VBulk", "3VGrad", "2V", "3VNoisy", "2VNoisy"]
def upload(vartype, algtype):
    if vartype == "Labelling proportion":
        if algtype == "3VBulk":
            filename = "3VBulk/3Vfixeddata.p"
        elif algtype == "3VGrad":
            filename = "3VGrad/3VGradfixeddata.p"
        elif algtype == "2V":
            filename = "2V/2Vfixeddata.p"
        xaxis = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    elif vartype == "Labelling size":
        if algtype == "3VBulk":
            filename = "3VBulk/3V100data.p"
        elif algtype == "3VGrad":
            filename = "3VGrad/3VGrad100data.p"
        elif algtype == "2V":
            filename = "2V/2V100data.p"
        xaxis = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    elif vartype == "Noise fixed 1.0":
        if algtype == "3VNoisy":
            filename = "3VNoisy/3VNoisyfixed_1.0data.p"
        elif algtype == "2VNoisy":
            filename = "2VNoisy/2VNoisyfixed_1.0data.p" 
        xaxis = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    elif vartype == "Noise 100 0.5":
        if algtype == "3VNoisy":
            filename = "3VNoisy/3VNoisy100_0.5data.p"
        elif algtype == "2VNoisy":
            filename = "2VNoisy/2VNoisy100_0.5data.p" 
        xaxis = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    data = pickle.load(open(filename,"rb"))
    algorithms = getalgs(algtype)
    print("Main variable: " + str(vartype))
    print("Algorithm types: " + str(algtype))
    print("")
    return data, algorithms, xaxis, algtype

# getalgs takes the initialisation algtype as input and returns the specific
# algorithms, to be used in the experiments, as output.       
def getalgs(algtype):
    if algtype == "3VBulk":
        algorithms = ['Arbitrary-Start', 'Full-Start', 'Batch']
    elif algtype == "3VGrad":
        algorithms = ['Online-Step', 'Online-Increment', 'Offline', 'Greedy']
    elif algtype == "2V":
        algorithms = ['Online 2V', 'Offline 2V']
    elif algtype == "3VNoisy":
        algorithms = ['N3V On-Naive', 'N3V On-Strict', 'N3V Off-Naive', 'N3V Off-Strict']
    elif algtype == "2VNoisy":
        algorithms = ['N2V Online', 'N2V Offline']
    return algorithms
    

# Mann-Whitney test to check if MCC scores can be said to belong to the same distribution.
# An important test to use when the data has been averaged in order to show that any differences
# are meaningful, in which the U value will indicate the hypothesis of belonging to the same
# distribution can be rejected at a confidence level as determined by alpha (but also revealed by p).
def xaxiseval(simpledata, algorithms, metrics, option):
    print("Metric: " + metrics[option])
    print("")
    simplexaxis = simpledata[option]
    for idx, i in enumerate(simplexaxis):
        for jidx in range(idx+1, len(algorithms)):
            stat, p = mannwhitneyu(i, simplexaxis[jidx])
            print(algorithms[idx] + ' vs ' + algorithms[jidx] + ' statistics = %.3f, p = %.3f' % (stat, p))
            # interpret
            alpha = 0.05
            if p > alpha:
            	print('Same distribution (fail to reject H0)')
            else:
            	print('Different distribution (reject H0)')
        print("")


# getsimpledata takes the loaded data set and returns the data points for each metric without
# reference to the x-axis.
def getsimpledata(data, algorithms):
    simplestats = []
    for metric in data:
        metricstat = []
        for item in range(len(algorithms)):
            metricstat.append([])
        for j in metric:
            for algidx, alg in enumerate(j):
                for sample in alg:
                    metricstat[algidx].append(sample)
        simplestats.append(metricstat)
    return simplestats


# getsimplestats takes the simple data (missing x-axis reference) for each metric
# and returns the mean and STDev for each algorithm.
def getsimplestats(simpledata, algorithms, metrics):
    simplestats = []
    # Iterate through metrics
    for idx, i in enumerate(simpledata):
        simplestats.append([])
        metricstr = metrics[idx]
        means = []
        stdvs = []
        if idx == 8:
            best = 999999.9
        else:
            best = -999999.9
        # Iterate through algorithms
        for jidx, j in enumerate(i):
            newmean = statistics.mean(j)
            newstdv = statistics.stdev(j)
            means.append(newmean)
            stdvs.append(newstdv)
            if idx == 8:
                if newmean < best:
                    best = newmean
            else:
                if newmean > best:
                    best = newmean
            simplestats[idx].append([newmean, newstdv])
        for kidx, k in enumerate(means):
            if k == best:
                metricstr += " & \\textbf{" + str(round(k, 3)) + "} & " + str(round(stdvs[kidx], 3))
            else:
                metricstr += " & " + str(round(k, 3)) + " & " + str(round(stdvs[kidx], 3))
        print(metricstr + " \\\ ")
    print("")
    return simplestats

# resultsMCC takes the MCC data including the x-axis reference and returns the mean and
# STDev for each algorithm.
def resultsxaxis(data, algorithms, xaxis, metrics, option):
    metric = metrics[option]
    print("Metric: " + metric)
    print("")
    xaxisstats = []
    means = []
    stdvs = []
    best = []
    xaxisdata = data[option]
    if max(xaxis) in [100, 0.5] or metric == 'Time Taken (secs)':
        bestbyprop = False
        for datapoint in xaxis:
            means.append([])
            stdvs.append([])
            if metric == 'MCC Score':
                best.append(-999999.9)
            elif metric == 'Time Taken (secs)':
                best.append(999999.9)
    elif max(xaxis) == 1.0:
        bestbyprop = True
        for alg in algorithms:
            means.append([])
            stdvs.append([])
            best.append(-999999.9)
    # Iterate through proportions
    for idx, i in enumerate(xaxisdata):
        xaxisstats.append([])
        # Iterate through algorithms
        for jidx, j in enumerate(i):
            newmean = statistics.mean(j)
            newstdv = statistics.stdev(j)
            if bestbyprop:
                means[jidx].append(newmean)
                stdvs[jidx].append(newstdv)
                if metric == 'MCC Score':
                    if newmean > best[jidx]:
                        best[jidx] = newmean
                elif metric == 'Time Taken (secs)':
                    if newmean < best[jidx]:
                        best[jidx] = newmean
            else:
                means[idx].append(newmean)
                stdvs[idx].append(newstdv)
                if metric == 'MCC Score':
                    if newmean > best[idx]:
                        best[idx] = newmean
                elif metric == 'Time Taken (secs)':
                    if newmean < best[idx]:
                        best[idx] = newmean
            xaxisstats[idx].append([newmean, newstdv])
    for kidx, k in enumerate(xaxis):
        axisstr = str(k)
        if bestbyprop:
            for lidx, l in enumerate(means):
                newval = l[kidx]
                if newval == best[lidx]:
                    axisstr += " & \\textbf{" + str(round(newval, 3)) + "}"
                else:
                    axisstr += " & " + str(round(newval, 3))
                axisstr += " & " + str(round(stdvs[lidx][kidx], 3))
        else:
            for lidx in range(len(algorithms)):
                newval = means[kidx][lidx]
                if newval == best[kidx]:
                    axisstr += " & \\textbf{" + str(round(newval, 3)) + "}"
                else:
                    axisstr += " & " + str(round(newval, 3))
                axisstr += " & " + str(round(stdvs[kidx][lidx], 3))
        print(axisstr + " \\\ ")
    print("")
    return xaxisstats


# boxwhisker takes as input the simpledata, algtype, vartype, and algorithms. 
# It outputs box and whisker plots to the appropriate directory.
def boxwhisker(simpledata, algtype, vartype, algorithms):
    mccntime = [7, 8]
    for metidx in mccntime:
        sequence = []
        for idx, i in enumerate(simpledata[metidx]):
            sequence.append(i)
            """
            for jidx, j in enumerate(i):
                if j > 2.5:
                    print((jidx+1) % 12)
            """
            #print("")
        fig, ax = plt.subplots(figsize=(10, 6))
        # Remove top and right border
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # Remove y-axis tick marks
        ax.yaxis.set_ticks_position('none')
        # Set metric-specific details
        if metidx == 7:
            ax.set_ylim([0.0, 1.0])
            ax.set_title('Distribution of algorithm MCC Scores')
            savename = algtype + "/" + algtype + "_BW_MCC_" + vartype + ".png"
        elif metidx == 8:
            #ax.set_ylim([0.0, 20.0])
            ax.set_title('Distribution of algorithm execution times (secs)')
            savename = algtype + "/" + algtype + "_BW_time_" + vartype + ".png"
        # Add major gridlines in the y-axis
        ax.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
        ax.boxplot(sequence, labels=algorithms, whis = 1.5)
        plt.savefig(savename)
        plt.show()
    return


# lineplots takes as input a dataset, algtype, vartype, algorithms, and xaxis.
# It outputs line plots to the appropriate directory.
def lineplots(data, algtype, vartype, algorithms, xaxis):
    mccntime = [7, 8]
    for metidx in mccntime:
        averages = []
        for item in range(len(algorithms)):
            averages.append([])
        for idx, i in enumerate(data[metidx]):
            for algidx, alg in enumerate(i):
                averages[algidx].append(statistics.mean(alg))
        fig, ax = plt.subplots(figsize=(10, 6))
        for seqidx, sequence in enumerate(averages):
            ax.plot(sequence, label=algorithms[seqidx])
        # Remove top and right border
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # Remove y-axis tick marks
        ax.yaxis.set_ticks_position('none')
        # Set metric-specific details
        if metidx == 7:
            if vartype == "Labelling proportion":
                ax.set_xlabel('Labelling proportion')
                ax.set_title('Mean algorithm MCC Score by labelling proportion')
            elif vartype == "Labelling size":
                ax.set_xlabel('No. of labellings in set')
                ax.set_title('Mean algorithm MCC Score by labelling set size')
            else:
                ax.set_xlabel('Noise proportion')
                ax.set_title('Mean algorithm MCC Score by noise proportion')
            ax.set_ylim([0.0, 1.0])
            savename = algtype + "/" + algtype + "_LP_MCC_" + vartype + ".png"
        elif metidx == 8:
            if vartype == "Labelling proportion":
                ax.set_xlabel('Labelling proportion')
                ax.set_title('Mean algorithm execution time (secs) by labelling proportion')
            elif vartype == "Labelling size":
                ax.set_xlabel('No. of labellings in set')
                ax.set_title('Mean algorithm execution time (secs) by labelling set size')
            else:
                ax.set_xlabel('Noise proportion')
                ax.set_title('Mean algorithm execution time (secs) by noise proportion')
            #ax.set_ylim([0.0, 3.0])
            savename = algtype + "/" + algtype + "_LP_time_" + vartype + ".png"     
        ax.set_ylim(bottom=0)
        plt.xticks([r for r in range(len(averages[0]))], xaxis)
        # Add major gridlines in the y-axis
        ax.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(savename, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.show()
    return



metrics = ['Accuracy', 'Defeat Precision', 'Defeat Recall', 'No-defeat Precision',\
            'No-defeat Recall', 'Defeat F1 Score', 'No-defeat F1 Score', 'MCC Score', 'Time Taken (secs)']

# vartype = ["Labelling proportion", "Labelling size", "Noise fixed 1.0", "Noise 100 0.5"]
vartype = "Noise fixed 1.0"

# algtype = ["3VBulk", "3VGrad", "2V", "3VNoisy", "2VNoisy"]
algtype = "3VNoisy"

# loadup takes arguments vartype and algtype.
# loadup outputs = [data, algorithms, xaxis, algtype]
#loadup = upload(vartype, algtype)

# Simple data processing and statistical outputs.
#simpledata = getsimpledata(loadup[0], loadup[1])
#simplestats = getsimplestats(simpledata, loadup[1], metrics)
#xaxisMW = xaxiseval(simpledata, loadup[1], metrics, 8)

# The more detailed x-axis related output for MCC scores.
#xaxisfull = resultsxaxis(loadup[0], loadup[1], loadup[2], metrics, 8)

# Plotting inputs = [data, algtype, vartype, algorithms]
#boxwhisker(simpledata, algtype, vartype, loadup[1])

# Plotting inputs = [data, algtype, vartype, algorithms]
#lineplots(loadup[0], algtype, vartype, loadup[1], loadup[2])

#plt.rcdefaults()
fig, ax = plt.subplots()
"""
xvals = [0.854,0.664,0.718,0.745,0.88,0.955,0.965,0.966,0.942,0.657,0.291,0.474]
yvals = ["Arbitrary-Start","Full-Start","Batch","Bulk mean","Online-Step",\
         "Online-Inc","Offline","Greedy","Gradualism mean","Online 2V",\
         "Offline 2V","2-valued mean"]
colours=['blue','blue','blue','black','green','green','green','green',\
       'black','red','red','black']
y_pos = np.arange(len(yvals))
ax.barh(y_pos, xvals, align='center', color=colours)
ax.set_yticks(y_pos)
ax.set_yticklabels(yvals)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_title('Mean MCC Scores for Favourable Labelling Parameters')
plt.savefig("Sigma_MCC_comparison.png", bbox_inches='tight')
"""
"""
xvals = [-0.112,-0.302,-0.248,-0.221,-0.086,-0.011,-0.001,0,-0.025,-0.309,\
         -0.675,-0.492,]
yvals = ["Arbitrary-Start","Full-Start","Batch","Bulk mean","Online-Step",\
         "Online-Inc","Offline","Greedy","Gradualism mean","Online 2V",\
         "Offline 2V","2-valued mean"]
colours=['blue','blue','blue','black','green','green','green','green',\
       'black','red','red','black']
y_pos = np.arange(len(yvals))
ax.set_xlim([-1.0,0.0])
ax.barh(y_pos, xvals, align='center', color=colours)
ax.set_yticks(y_pos)
ax.set_yticklabels(yvals)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_title('Difference from Best Mean MCC Score for Favourable Labelling Parameters')
plt.savefig("Sigma_MCC_difference.png", bbox_inches='tight')
"""
"""
xvals = [0.323, 0.429, 0.354, 0.432, 0.385, 0.165, 0.033, 0.099]
yvals = ["N3V On-Naive", "N3V On-Strict", "N3V Off-Naive", "N3V Off-Strict",\
         "3-valued mean", "N2V Online", "N2V Offline", "2-valued mean"]
colours=['green','green','green','green', 'black', 'red','red','black']
y_pos = np.arange(len(yvals))
ax.set_xlim([0.0,1.0])
ax.barh(y_pos, xvals, align='center', color=colours)
ax.set_yticks(y_pos)
ax.set_yticklabels(yvals)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_title('Mean MCC Scores for Favourable Labelling Parameters')
plt.savefig("Noisy_MCC_comparison.png", bbox_inches='tight')
"""
xvals = [-0.109, -0.003, -0.078, 0, -0.047, -0.267, -0.399, -0.333]
yvals = ["N3V On-Naive", "N3V On-Strict", "N3V Off-Naive", "N3V Off-Strict",\
         "3-valued mean", "N2V Online", "N2V Offline", "2-valued mean"]
colours=['green','green','green','green', 'black', 'red','red','black']
y_pos = np.arange(len(yvals))
ax.set_xlim([-1.0, 0.0])
ax.barh(y_pos, xvals, align='center', color=colours)
ax.set_yticks(y_pos)
ax.set_yticklabels(yvals)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_title('Difference from Best Mean MCC Score for Favourable Labelling Parameters')
plt.savefig("Noisy_MCC_difference.png", bbox_inches='tight')