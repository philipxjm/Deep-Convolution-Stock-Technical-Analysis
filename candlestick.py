import os
import math
import random
import numpy as np
import csv
import plotly
import plotly.plotly as py
import datetime
from plotly.tools import FigureFactory as FF
from datetime import datetime
import pandas.io.data as web

class Candlestick(object):
    """docstring for Candlestick"""
    def __init__(self):
        self.table = [];
        self.openPrice = [];
        self.highPrice = [];
        self.lowPrice = [];
        self.closePrice = [];
        self.dates = [];

    def importCSV(self, csvURL):
        with open(csvURL, 'rb') as f:
            reader = csv.reader(f, delimiter=',');
            data = list(reader);
            return data;

    def makeList(self, index, li):
        l = [];
        for x in xrange(0,len(li)):
            l.append(float(li[x][index]));
        return l;

    def makeDateList(self, index, li):
        l = [];
        for x in xrange(0,len(li)):
            d = datetime.strptime(li[x][index], "%Y%m%d").date()
            l.append(d);
        return l;

    def makeCandlestick(self, filename):
        self.table = self.importCSV(filename);
        self.dates = self.makeDateList(0, self.table);
        self.openPrice = self.makeList(2, self.table);
        self.highPrice = self.makeList(3, self.table);
        self.lowPrice = self.makeList(4, self.table);
        self.closePrice = self.makeList(5, self.table);

        fig = FF.create_candlestick(self.openPrice, self.highPrice, self.lowPrice, self.closePrice, dates=self.dates);
        py.plot(fig, filename='finance/aapl-candlestick', validate=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Candlestick Chart Construction Using CSV");
    parser.add_argument('filename', type=str, help="CSV File to Import (Comma seperated, format: [Date, Time, Open, High, Low, Close, Volume]");
    opts = parser.parse_args();

    candlestick = Candlestick();
    candlestick.makeCandlestick(opts.filename);
