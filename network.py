#! /usr/bin/env python

# neural network with flexible node configurations
# author: Philip Xu

import os
import math
import random
import numpy as np
import csv
import binascii
import time
import sys
from progressbar import Bar, Percentage, ProgressBar, SimpleProgress

random.seed(0);

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a) * random.random() + a;

# Make a matrix
def makeMatrix(I, J, fill=0.0):
    m = [];
    for i in range(I):
        m.append([fill]*J);
    return m;

# sigmoid function, uses tanh
def sigmoid(x):
    return math.tanh(x);

# derivative of sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2;

class Network:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1; # +1 for bias node
        self.nh = nh + 1; # +1 for bias node
        self.no = no;

        # activations for nodes
        self.ai = [1.0]*self.ni;
        self.ah = [1.0]*self.nh;
        self.ao = [1.0]*self.no;

        # create weights matrices
        self.wi = makeMatrix(self.ni, self.nh);
        self.wo = makeMatrix(self.nh, self.no);
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.5, 0.5);
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-0.1, 0.1);

        # last change in weights for momentum
        self.ci = makeMatrix(self.ni, self.nh);
        self.co = makeMatrix(self.nh, self.no);

        # parameters
        self.I=1; # iteration
        self.N=0.00001; # learning
        self.M=0.001; # momentum
        self.debug = False; # instead of recognizing and writing to a file, debug will analyze the accuracy
        self.outputWeightFilename = "default"; # output filename of weight files
        self.outputRecognitionFilename = "default.txt"; # output filename
        self.trainingData = None; # training data
        self.testingData = None; # testing data
        self.trainingProgress = 0; # progress of training
        self.trainingTotal = 0; # total numbers of times the network needs to be trained, = I * len(trainingData)
        self.recognitionProgress = 0; # progress
        self.recognitionTotal = 0; # total task
        self.outlist = []; # list of output;

    # return outlist
    def getOutlist(self):
        return self.outlist;

    # set learning constant, default 0.00001
    def setLearning(self, N):
        self.N = N;

    # set momentum constant, default 0.001
    def setMomentum(self, M):
        self.M = M;

    # set iteration, default 1
    def setIteration(self, I):
        self.I = I;

    # set weights to wi, wo matrice
    def setWeights(self, wi, wo):
        self.wi = wi;
        self.wo = wo;

    # set output weight file URL
    def setOutputWeightFileName(self, filename):
        self.outputWeightFilename = filename;

    # set output recognized file URL
    def setOutputRecognitionFileName(self, filename):
        self.outputRecognitionFilename = filename;

    # enable debug
    def enableDebugRecognition(self):
        self.debug = True;

    # set the training data to a matrix
    def setTrainingData(self, data):
        self.trainingData = data;

    # set the testing data to a matrix
    def setTestingData(self, data):
        self.testingData = data;

    # feed forward data
    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs');

        # input activations
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(float(inputs[i]))
            self.ai[i] = float(inputs[i]);
        self.ai.append(float(1.0));

        # hidden activations
        for j in range(self.nh):
            sum = 0.0;
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j];
            self.ah[j] = sigmoid(sum);

        # output activations
        for k in range(self.no):
            sum = 0.0;
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k];
            #print("sigmoid: " + str(sigmoid(sum)));
            self.ao[k] = sigmoid(sum);

        return self.ao[:];

    # backpropagate errors
    def backPropagate(self, tar, N, M):
        targets = [];
        for t in tar:
            targets.append((float(t)*2) - 1);
        if len(targets) != self.no:
            raise ValueError('wrong number of target values');

        # calculate error terms for output
        output_deltas = [0.0] * self.no;
        for k in range(self.no):
            error = float(targets[k])-self.ao[k];
            output_deltas[k] = dsigmoid(self.ao[k]) * error;

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh;
        for j in range(self.nh):
            error = 0.0;
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k];
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error;

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j];
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k];
                self.co[j][k] = change;
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i];
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j];
                self.ci[i][j] = change;

        # calculate error
        error = 0.0;
        for k in range(len(targets)):
            error = error + 0.5*(float(targets[k])-self.ao[k])**2;
        return error;

    # print weights matrices
    def weights(self):
        print('Input weights:');
        print(self.wi);
        print('\nOutput weights:');
        print(self.wo);

    # save weight matrices as csv
    def saveWeights(self):
        # save matrices to specified location
        if((self.outputWeightFilename is not None)):
            newpath = "weights/" + self.outputWeightFilename;
            if not os.path.exists(newpath):
                os.makedirs(newpath);
            np.savetxt("weights/" + self.outputWeightFilename + "/wi.csv", self.wi, delimiter=" ");
            np.savetxt("weights/" + self.outputWeightFilename + "/wo.csv", self.wo, delimiter=" ");
            print("\nSaved weight matrices in: " + "weights/" + self.outputWeightFilename + "/");

        # save matrices to default folder
        else:
            newpath = "weights/defaultweights";
            if not os.path.exists(newpath):
                os.makedirs(newpath);
            np.savetxt("weights/defaultweights/wi.csv", self.wi, delimiter=" ");
            np.savetxt("weights/defaultweights/wo.csv", self.wo, delimiter=" ");
            print("\nSaved weight matrices in: " + "weights/defaultweights/");

    # read weight csv into matrices
    def importWeights(self, wiURL, woURL):
        self.wi = np.loadtxt(open(wiURL,"rb"),delimiter=" ");
        self.wo = np.loadtxt(open(woURL,"rb"),delimiter=" ");

    # training sequence
    def train(self):
        # N: learning rate
        # M: momentum factor
        if self.trainingData is None:
            print "No Training Data Loaded";
            return;

        start = time.time();
        print("Training Progress: ");
        self.trainingProgress = 0;
        self.trainingTotal = self.I * len(self.trainingData);
        pbar = ProgressBar(widgets=[Percentage(), Bar(), SimpleProgress()], maxval=int(self.trainingTotal)).start();

        for i in range(int(self.I)):
            for x in range(len(self.trainingData)):
                error = 0.0;
                inputs = self.trainingData[x][1:];

                # target output node activations
                targets = self.fromCharacterToBinary(self.trainingData[x][0]);

                # feedforward inputs
                self.update(inputs);

                # generate new Learning Constant
                # self.setLearning(self.logisticFunction(100.0*(float(self.trainingProgress)/float(self.trainingTotal))));

                # backpropagate errors
                error = error + self.backPropagate(targets, self.N, self.M);

                # update progress
                self.trainingProgress += 1;

                pbar.update(float(n.trainingProgress));

        pbar.finish();
        end = time.time();
        print("\nTime took to train: " + str(end - start) + " seconds");

    def logisticFunction(self, x):
        y = (float(-0.1)/float(1.0+math.e**(-0.5*x)))+0.1;
        if y < 0.00001:
            return 0.00001;
        else:
            return y;

# imports csv as list
def importCSV(csvURL):
    with open(csvURL, 'rb') as f:
        reader = csv.reader(f, delimiter=' ');
        trainingData = list(reader);
        return trainingData;

# move console cursor up
def up():
    sys.stdout.write('\x1b[1A')
    sys.stdout.flush()

# move console cursor down
def down():
    sys.stdout.write('\n')
    sys.stdout.flush()

if __name__ == '__main__':
    # adding arguments
    import argparse
    parser = argparse.ArgumentParser(description="Neural Network for Stock Exchange Prediction");
    parser.add_argument('--train', type=str, help="train this network, set name of training csv file");
    parser.add_argument('--write-weights', type=str, help="set prefered names of directory to save weights");
    parser.add_argument('--read-weights', type=str, help="set the input weights of this network, set name of directory to read weights");
    parser.add_argument('--set-iteration', type=float, default=1, help="iteration number, default value 1");
    parser.add_argument('--set-learning', type=float, default=0.00001, help="learning constant, default value 0.00001");
    parser.add_argument('--set-momentum', type=float, default=0.001, help="momentum constant, default value 0.001");
    parser.add_argument('--enable-debug', action="store_true", default=False, help="whether to recognize using debug algorithm or not");
    parser.add_argument('--set-output', type=str, default="default.txt", help="set name of output file");
    opts = parser.parse_args();

    # creating a network with input, hidden, and output nodes
    n = Network(26, 150, 7); # To Be Changed

    # setting data vars
    trainingData = None;
    testingData = None;

    # setting vars according to args
    if(opts.set_iteration is not None):
        n.setIteration(opts.set_iteration);

    if(opts.set_learning is not None):
        n.setLearning(opts.set_learning);

    if(opts.set_momentum is not None):
        n.setMomentum(opts.set_momentum);

    if(opts.enable_debug):
        n.enableDebugRecognition();

    if((opts.write_weights is not None)):
        n.setOutputWeightFileName(opts.write_weights);

    if((opts.set_output is not None)):
        n.setOutputRecognitionFileName(opts.set_output);

    if((opts.read_weights is not None)):
        n.importWeights("weights/" + opts.read_weights + "/wi.csv", "weights/" + opts.read_weights + "/wo.csv");

    if(opts.train is not None):
        trainingData = importCSV(opts.train);
        n.setTrainingData(trainingData);

    # train
    if((trainingData is not None)):
        n.train();
        n.saveWeights();
