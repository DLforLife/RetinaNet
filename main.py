"""
This is the main API for instantiating the network. It mainly:
- parses the configurations file
- loads the model
- instantiates the trainer object passing the required configs
"""
from common.utils.utils import *
from src import *

def main():
	config_args = parse_args()
	folders_list = create_experiment_dirs(config_args.exp_dir)
	model=class_by_name(config_args.model)
	agent=class_by_name(config_args.agent)
	agent=agent()
	#######
	#todo :
	#setup the agent structure so we can pass the model to it
	#expected api :
	# agent=agent(config=config_args,folders=folders_list,model_used=model)
	#######

	try:
		agent.run()
		agent.finish()
	except KeyboardInterrupt :
		agent.finish()

if __name__ == '__main__':
	main()


