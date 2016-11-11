#! /usr/bin/env python
from rnn import RNN
import csv
import os

class Reader:
	"""Reads csv input files"""
	def __init__(self):
		pass

	# Consumes a filename and returns a list of all stock data contained in that file.
	def readCSV(self, filename):
		with open(filename, "rb") as csvfile:
			readr = csv.reader(csvfile, delimiter = ",", quotechar = "|")
			data = []
			for row in readr:
				data.append(map(lambda x: float(x), row))
			return data

	# Consumes a filename, and two ints. Returns a list of any stock values that is between these two dates.
	def readCSVYears(self, filename, startYear = 1900, endYear = 2020):
		return [x for x in self.readCSV(filename) if int(str(x[0])[:4]) >= startYear and int(str(x[0])[:4]) <= endYear]

	# Consumes a filename, and a list, and returns null. Creates a csv file with the file name with all the elements in the list.
	def writeCSV(self, filename, dataset):
		with open(filename, 'wb') as csvfile:
			csv_writer = csv.writer(csvfile, delimiter=',');
			for data in dataset:
				csv_writer.writerow(data);


reader = Reader()
reader.writeCSV("test.csv", reader.readCSVYears("data/daily/table_a.csv", 2011, 2012))