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
	config_args	= create_experiment_dirs(config_args)

	model_class=class_by_name(config_args.model)
	model=model_class(config_args)

	agent_class=class_by_name(config_args.agent)
	agent=agent_class(config_args,model)

	try:
		agent.run()
		agent.finish()
	except KeyboardInterrupt :
		agent.finish()

if __name__ == '__main__':
	main()


