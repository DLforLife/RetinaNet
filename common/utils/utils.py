from bunch import Bunch
import os
import sys
import argparse
import json
import inspect

def parse_args():
	"""
	Parse the arguments of the program
	:return: (config_args)
	:rtype: tuple
	"""
	# Create a parser
	parser = argparse.ArgumentParser(description="")
	parser.add_argument('--config', default=None, type=str, help='Configuration file')
	# Parse the arguments
	args = parser.parse_args()
	# parse the configurations from the config json file provided
	with open(args.config, 'r') as config_file:
		config_args_dict = json.load(config_file)
	# convert the dictionary to a namespace using bunch lib
	config_args = Bunch(config_args_dict)

	print(config_args)
	return config_args

def create_experiment_dirs(args):
	args,dirs=update_configs(args)
	try:
		for dir_ in dirs:
			if not os.path.exists(dir_):
				os.makedirs(dir_)
		print("Experiment directories created!")
		# return experiment_dir, summary_dir, checkpoint_dir, output_dir, test_dir
		return args
	except Exception as err:
		print("Creating directories error: {0}".format(err))
		exit(-1)
def update_configs(args):
	args.experiment_dir = os.path.realpath("experiments/" + args.exp_dir + "/")
	args.summary_dir = args.experiment_dir + '/summaries/'
	args.checkpoint_dir = args.experiment_dir + '/checkpoints/'
	args.output_dir = args.experiment_dir + '/outputs/'
	dirs = [args.summary_dir, args.checkpoint_dir, args.output_dir]
	return args,dirs

def class_by_name(model):
	module=None
	for i in sys.modules.keys():
		try:
			module_under_test =getattr(sys.modules[i],model)
			if inspect.isclass(module_under_test):
				module = module_under_test
		except:
			pass
	if (not inspect.isclass(module) ) :
		print("[error]",model,"don't exist please check it's name and location")
		exit(-1)
	return (module)

