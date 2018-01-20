import tensorflow as tf
import numpy as np
from src.agents.basic_agent import BasicAgent

class MainAgent(BasicAgent):
	def __init__(self, config,model):
		print("\n We are in the ChildAgent\n")
		BasicAgent.__init__(self,config=config,model=model)
	def run(self):
		self.train()
	def finish(self):
		pass

	def train(self):
		self.nb_epochs = self.config.max_epoch
		self.current_epoch = self.model.global_epoch_tensor.eval(self.sess)+1
		summaries_dict = dict()
		for i in range(self.current_epoch,self.nb_epochs,1):
			loss,accuracy=self.train_epoch()
			total_loss = np.mean(loss)
			total_acc = np.mean(accuracy)
			summaries_dict['Loss'] = total_loss
			summaries_dict['Accuracy'] = total_acc
			self.add_summary(i,type='train',summaries_dict=summaries_dict)

	def train_epoch(self):
		data_len = self.x_train_len
		nb_batches = data_len/self.config.batch_size
		loss_list = []
		acc_list = []
		for i in range(nb_batches):
			loss,acc =self.train_step()
			loss_list += [loss]
			acc_list += [acc]
		return loss_list,acc_list
	def validate(self):
		pass
	def test(self):
		pass


if __name__ == '__main__':
	agent= MainAgent()
