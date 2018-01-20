import tensorflow as tf
import numpy as np
from src.agents.basic_agent import BasicAgent
from src.agents.dataset_agent import dataset_loader
class MainAgent(BasicAgent):
	def __init__(self, config,model):
		print("\n We are in the ChildAgent\n")
		BasicAgent.__init__(self,config=config,model=model)
		self.dataset = dataset_loader(config=config)

	def run(self):
		if self.config.mode == 'train':
			self.train()
		else :
			print('other modes is not yet done')

	def finish(self):
		print("[info] Finalizing   .... ")
		print("[info] Saving model .... ")
		self.save()
		self.train_summary_writer.flush()
		self.valid_summary_writer.flush()
		self.sess.close()

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
			if i % self.config.test_every == 0 :
				self.validate(i)

	def train_epoch(self):
		data_len = self.x_train_len
		nb_batches = data_len/self.config.batch_size
		loss_list = []
		acc_list = []
		for i in range(nb_batches):
			loss,acc =self.do_batch(is_trainig=True)
			loss_list += [loss]
			acc_list += [acc]
		return loss_list,acc_list

	def do_batch(self,is_trainig):
		feed_dict= self.dataset.next_batch(training=is_trainig)
		_,loss, acc = self.sess.run(
			[self.model.train_op,self.model.loss,self.model.train_accuracy],
			feed_dict=feed_dict)

	def validate(self,current_epoch):
		data_len = self.x_train_len
		nb_batches = data_len / self.config.batch_size
		loss_list = []
		acc_list = []
		summaries_dict = dict()
		for i in range(nb_batches):
			loss, accuracy = self.do_batch(is_trainig=False)
			loss_list += [loss]
			acc_list += [accuracy]

		total_loss = np.mean(loss)
		total_acc = np.mean(accuracy)
		summaries_dict['Loss'] = total_loss
		summaries_dict['Accuracy'] = total_acc
		self.add_summary(current_epoch, type='valid', summaries_dict=summaries_dict)

	def test(self):
		pass


if __name__ == '__main__':
	agent= MainAgent()
