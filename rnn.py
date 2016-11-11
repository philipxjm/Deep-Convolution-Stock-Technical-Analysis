#! /usr/bin/env python

import os
import math
import random
import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer
from pybrain.supervised import RPropMinusTrainer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
import matplotlib.pyplot as plt
import sys

class RNN:
	"""An object for recurrent neural networks using PyBrain"""
	def __init__(self, ni, nh, no):
		self.ni = ni
		self.nh = nh
		self.no = no
		self.n = buildNetwork(self.ni, self.nh, self.no, hiddenclass = LSTMLayer, outputbias=True, recurrent=True)

	def printParams(self):
		for mod in self.n.modules:
			print("Module:", mod.name)
			if mod.paramdim > 0:
				print("--parameters:", mod.params)
			for conn in self.n.connections[mod]:
				print("-connection to", conn.outmod.name)
				if conn.paramdim > 0:
					print("- parameters", conn.params)
		if hasattr(self.n, "recurrentConns"):
			print("Recurrent connections")
			for conn in self.n.recurrentConns:
				print("-", conn.inmod.name, " to", conn.outmod.name)
				if conn.paramdim > 0:
					print("- parameters", conn.params)

	def saveNetwork(self, filename):
		NetworkWriter.writeToFile(self.n, filename)

	def loadNetwork(self, filename):
		self.n = NetworkReader.readFrom(filename) 

	def train(self, ds, epochs_per_cycle, cycles):
		trainer = RPropMinusTrainer(self.n, dataset=ds)
		train_errors = []
		for i in xrange(cycles):
			trainer.trainEpochs(epochs_per_cycle)
			train_errors.append(trainer.testOnData())
			epoch = (i+1) * epochs_per_cycle
			print("\r epoch {}/{}".format(epoch, epochs_per_cycle * cycles))
			sys.stdout.flush()
		print("Final Error: " + str(train_errors[-1]))
		return train_errors[-1]

	def plotErrors(self, epochs_per_cycle, cycles, errors):
		plt.plot(range(0, epochs_per_cycle * cycles, epochs_per_cycle), errors)
		plt.xlabel("epoch")
		plt.ylabel("error")
		plt.show()

	def activate(self, inputs):
		return self.n.activate(inputs)