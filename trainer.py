#! /usr/bin/env python
from rnn import RNN
from reader import Reader
import csv
from pybrain.datasets import SequentialDataSet

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description="Trainer for LSTM Net");
	parser.add_argument('filename', type=str, help="input training data");
	parser.add_argument('--epochs-per-cycle', type=int, default=10, help="epochs per cycle, default value 10")
	parser.add_argument('--cycles', type=int, default=10, help="amount of cycles, default value 10")
	parser.add_argument('--learning', type=float, default=0.01, help="learning constant, default value 0.01")
	parser.add_argument('--debug', action="store_true", default=False, help="whether to recognize using debug algorithm or not")
	parser.add_argument("--input-nodes", type=int, default=5, help="number of input channels to the LSTM")
	parser.add_argument("--hidden-nodes", type=int, default=20, help="number of hidden channels to the LSTM")
	parser.add_argument("--output-nodes", type=int, default=1, help="number of output channels to the LSTM")
	parser.add_argument('--output', type=str, default="default.xml", help="set name of output file to store trained network");
	parser.add_argument('--read-weights', type=str, help="set the input weights of this network, set name of directory to read weights");
	opts = parser.parse_args();

	ds = SequentialDataSet(opts.input_nodes, opts.output_nodes)
	data = []
	with open(opts.filename, "rb") as csvfile:
		readr = csv.reader(csvfile, delimiter = ",")
		for row in readr:
			data.append(map(lambda x: float(x), row))
	for i in xrange(0, len(data)-1):
		ds.addSample(data[i][2:], data[i+1][2])


	net = RNN(opts.input_nodes, opts.hidden_nodes, opts.output_nodes)
	# if opts.debug:
	# 	print(net.n.module.indim)
	net.plotErrors(opts.epochs_per_cycle, opts.cycles, net.train(ds, opts.epochs_per_cycle, opts.cycles))
	net.saveNetwork(opts.output)