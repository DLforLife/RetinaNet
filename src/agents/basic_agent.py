import tensorflow as tf
from src import *

class BasicAgent:
	def __init__(self, config, model):
		self.model = model
		self.config = config
		print("\n We are in the MainAgent\n")
		# Reset the graph
		tf.reset_default_graph()
		# Create the session of the graph
		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth = True
		config.gpu_options.per_process_gpu_memory_fraction = self.config.gpu_factor
		#create Session using the config files
		self.sess = tf.Session(config=config)
		# build the model
		self.model.build()
		# init the summaries
		self.summary_placeholders = {}
		self.summary_ops = {}
		self.summary_tags = []
		self.scalar_summary_tags = ['Loss','Accuracy','Mean_IOU']
		self.images_summary_tags = []
		self.train_summary_writer = tf.summary.FileWriter(self.config.summary_dir+'train/', self.sess.graph)
		self.valid_summary_writer = tf.summary.FileWriter(self.config.summary_dir+'valid/', self.sess.graph)
		self.init_summaries()

		# To initialize all variables
		self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		self.sess.run(self.init)

		self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep,
									save_relative_paths=True,
									allow_empty=True)

	def init_summaries(self):

		with tf.variable_scope('summary'):
			for tag in self.scalar_summary_tags:
				self.summary_tags += tag
				self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
				self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
			for tag, shape in self.images_summary_tags:
				self.summary_tags += tag
				self.summary_placeholders[tag] = tf.placeholder('float32', shape, name=tag)
				self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag], max_outputs=10)


	def add_summary(self, step,type='train', summaries_dict=None, summaries_merged=None):
		"""
		Add the summaries to tensorboard
		:param step:
		:param type:
		:param summaries_dict:
		:param summaries_merged:
		:return:
		"""
		if summaries_dict is not None:
			summary_list = self.sess.run([self.summary_ops[tag] for tag in summaries_dict.keys()],
										 {self.summary_placeholders[tag]: value for tag, value in
										  summaries_dict.items()})
			for summary in summary_list:
				if type == 'train':
					self.train_summary_writer.add_summary(summary, step)
				else:
					self.valid_summary_writer.add_summary(summary, step)
		if summaries_merged is not None:
			if type == 'train':
				self.train_summary_writer.add_summary(summaries_merged, step)
			else:
				self.valid_summary_writer.add_summary(summaries_merged, step)

	def load(self, best=False):
		"""
		Load the latest checkpoint
		:return:
		"""
		if not best:
			latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
		else :
			latest_checkpoint = tf.train.latest_checkpoint(self.config.best_checkpoint_dir)
		if latest_checkpoint:
			print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
			self.saver.restore(self.sess, latest_checkpoint)
			if best:
				print("Model loaded from the best checkpoint")
			else:
				print("Model loaded from the latest checkpoint")
		else:
			if best :
				print('failed to find best checkpoint exiting ..')
				exit(-1)
			else:
				print("\n First time to train ..\n")

	def save(self, best=False):
		if not best:
			# print("[info]saving model....")
			self.saver.save(self.sess, self.config.checkpoint_dir, self.model.global_step_tensor)
		else:
			print("saving a checkpoint for the best model")
			self.saver.save(self.sess, self.config.best_checkpoint_dir, self.model.global_step_tensor)
			print("[info]Saved a checkpoint for the best model successfully")

