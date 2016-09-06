import os
import time


class Logger:
	def __init__(self, debug=False):
		self.debug = debug
		if self.debug:
			return  # Do not create folders if in debug mode

		output_folder = './output/'
		run_folder = 'run%Y%m%d-%H%M%S/'
		if not os.path.exists(output_folder):
			os.makedirs(output_folder)
		self.path = ''.join([output_folder, time.strftime(run_folder)])
		if not os.path.exists(self.path):
			os.makedirs(self.path)

	def log(self, data):
		if self.debug:
			return
		try:
			logfile = open(self.path + 'log.txt', 'a')
		except IOError:
			print 'Logger:log IO error while opening log file'
			return
		if type(data) is dict:
			for k in data:
				logfile.write(str(k) + ': ' + str(data[k]) + '\n')
				print str(k) + ': ' + str(data[k])
		if type(data) is tuple:
			logfile.write(str(data[0]) + ': ' + str(data[1]) + '\n')
		if type(data) is str:
			logfile.write(data + '\n')
			print data

	def to_csv(self, filename, row):
		if self.debug:
			return
		try:
			f = open(self.path + filename, 'a')
		except IOError:
			print 'Logger:to_csv IO error while opening file'
			return
		string = ','.join([str(val) for val in row])
		string = string + '\n' if not string.endswith('\n') else ''
		f.write(string)
		f.close()

