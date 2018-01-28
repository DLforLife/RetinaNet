import tensorflow as tf
import numpy as np
from tqdm import *
from src.agents.basic_agent import BasicAgent
from src.agents.dataset_agent import dataset_loader
class MainAgent(BasicAgent):
	def __init__(self, config,model):
		print("\n We are in the ChildAgent\n")
		BasicAgent.__init__(self,config=config,model=model)
		self.init_generators()

	def init_generators(self):
		self.train_generator = dataset_loader(config=self.config,x=self.config.x_train,y=self.config.y_train,augment=True)
		self.valid_generator = dataset_loader(config=self.config,x=self.config.x_val,y=self.config.y_val,augment=False)
		#self.test_generator  = dataset_loader(config=self.config,x=self.config.x_test,y=self.config.y_test,augment=False)

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
		self.nb_epochs = self.config.epochs
		self.current_epoch = self.model.global_epoch_tensor.eval(self.sess)+1
		summaries_dict = dict()
		for i in range(self.current_epoch,self.nb_epochs,1):
			print('starting training  epoch ',i)
			loss,accuracy=self.train_epoch()
			total_loss = np.mean(loss)
			summaries_dict['Loss'] = total_loss
			self.add_summary(i,type='train',summaries_dict=summaries_dict)
			if i % self.config.test_every == 0 :
				self.validate(i)

	def train_epoch(self):
		data_len = self.train_generator.data_len
		nb_batches = data_len // self.config.batch_size
		loss_list = []
		acc_list = []
		for i in tqdm(range(10)):
			loss = self.do_batch(batch_type='train')
			loss_list += [loss]
		print(np.mean(loss_list))
		return loss_list, acc_list

	def do_batch(self,batch_type):
		if batch_type =='train':
			is_training=True
			inputs,targets= self.train_generator.next_batch()
		elif batch_type  == 'valid':
			is_training = False
			inputs,targets= self.valid_generator.next_batch()
		else:
			is_training = False
			inputs,targets= self.test_generator.next_batch()

		feed_dict = {self.model.input: inputs,
					 self.model.target: targets,
					 self.model.is_training: is_training}
		return_arr = []
		if batch_type =='train':
			_, loss = self.sess.run(
				[self.model.train_op, self.model.loss],
				feed_dict=feed_dict)
			return_arr=[loss]
		elif batch_type  == 'valid':
			loss = self.sess.run(
				[self.model.loss],
				feed_dict=feed_dict)
			return_arr=[loss]

		else:
			grid_out = self.sess.run([self.model.grid_output],feed_dict=feed_dict)
			return_arr=[grid_out]
		return return_arr

	def validate(self,current_epoch):
		print('starting validation epoch :',current_epoch)
		data_len = self.valid_generator.data_len
		nb_batches = data_len // self.config.batch_size
		loss_list = []
		acc_list = []
		summaries_dict = dict()
		for i in tqdm(range(nb_batches)):
			loss = self.do_batch(batch_type='valid')
			loss_list += [loss]
		total_loss = np.mean(loss_list)
		summaries_dict['Loss'] = total_loss
		self.add_summary(current_epoch, type='valid', summaries_dict=summaries_dict)#,summaries_merged=image_summary)

	def test(self):
		pass


if __name__ == '__main__':
	agent= MainAgent()
