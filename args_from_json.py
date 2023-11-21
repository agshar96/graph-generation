import json

### program configuration
class Args_from_json():
    def __init__(self, jsonFile):
        file = open(jsonFile)
        data = json.load(file)

        ### if clean tensorboard
        self.clean_tensorboard = data['clean_tensorboard']
        ### Which CUDA GPU device is used for training
        self.cuda = data['cuda']

        ### Which GraphRNN model variant is used.
        # The simple version of Graph RNN
        # self.note = 'GraphRNN_MLP'
        # The dependent Bernoulli sequence version of GraphRNN
        self.note = data['note']

        ## for comparison, removing the BFS compoenent
        # self.note = 'GraphRNN_MLP_nobfs'
        # self.note = 'GraphRNN_RNN_nobfs'

        ### Which dataset is used to train the model
        # self.graph_type = 'DD'
        # self.graph_type = 'caveman'
        # self.graph_type = 'caveman_small'
        # self.graph_type = 'caveman_small_single'
        # self.graph_type = 'community4'
        # self.graph_type = 'grid'
        # self.graph_type = 'grid_small'
        # self.graph_type = 'ladder_small'

        # self.graph_type = 'enzymes'
        # self.graph_type = 'enzymes_small'
        # self.graph_type = 'barabasi'
        # self.graph_type = 'barabasi_small'
        # self.graph_type = 'citeseer'
        # self.graph_type = 'citeseer_small'
        ## ARGOVERSE
        self.graph_type = data['graph_type']

        # self.graph_type = 'barabasi_noise'
        # self.noise = 10
        #
        # if self.graph_type == 'barabasi_noise':
        #     self.graph_type = self.graph_type+str(self.noise)

        # if none, then auto calculate
        self.max_num_node = None # max number of nodes in a graph
        self.max_prev_node = None # max previous node that looks back

        ### network config
        ## GraphRNN
        if 'small' in self.graph_type:
            self.parameter_shrink = 2
        else:
            self.parameter_shrink = 1
        self.hidden_size_rnn = int(128/self.parameter_shrink) # hidden size for main RNN
        self.hidden_size_rnn_output = 16 # hidden size for output RNN
        self.embedding_size_rnn = int(64/self.parameter_shrink) # the size for LSTM input
        self.embedding_size_rnn_output = 8 # the embedding size for output rnn
        self.embedding_size_output = int(64/self.parameter_shrink) # the embedding size for output (VAE/MLP)

        self.batch_size = data['batch_size'] # normal: 32, and the rest should be changed accordingly
        self.test_batch_size = data['test_batch_size']
        self.test_total_size = data['test_total_size'] #1000 Earlier
        self.num_layers = data['num_layers']

        ### training config
        self.num_workers = data['num_workers'] # num workers to load data, default 4
        self.batch_ratio = data['batch_ratio'] # how many batches of samples per epoch, default 32, e.g., 1 epoch = 32 batches
        self.epochs = data['epochs'] # now one epoch means self.batch_ratio x batch_size
        self.epochs_test_start = data['epochs_test_start'] #100 Earlier
        self.epochs_test = data['epochs_test'] #100 Earlier
        self.epochs_log = data['epochs_log'] #100
        self.epochs_save = data['epochs_save']

        self.lr = data['lr']
        self.milestones = data['milestones']
        self.lr_rate = data['lr_rate']

        self.sample_time = data['sample_time'] # sample time in each time step, when validating

        ### output config
        # self.dir_input = "/dfs/scratch0/jiaxuany0/"
        self.dir_input = "./"
        self.model_save_path = self.dir_input+data['model_save_path'] + '/' # only for nll evaluation
        self.graph_save_path = self.dir_input+data['graph_save_path'] + '/'
        self.figure_save_path = self.dir_input+data['figure_save_path'] + '/'
        self.figure_save_name = data['figure_save_name']
        self.timing_save_path = self.dir_input+ data['timing_save_path'] + '/'
        self.figure_prediction_save_path = self.dir_input+ data['figure_prediction_save_path'] + '/'
        self.nll_save_path = self.dir_input+ data['nll_save_path']+'/'


        self.load = data['load'] # if load model, default lr is very low
        self.load_epoch = data['load_epoch']
        self.save = data['save']
        # Vanilla graphRNN ignores node embedding, but if this is set to true we can integrate it into the model.
        ## ARGOVERSE
        self.use_node_embedding = data['use_node_embedding']
        self.node_embedding_size = data['node_embedding_size']
        ## ARGOVERSE -- These arguments only work with node embeddings
        self.use_small_graph = data['use_small_graph']
        self.small_node_num = data['small_node_num']
        if 'argoverse' in self.graph_type:
            self.use_embed_offset = data['use_embed_offset']
            self.dataset_file_name = data['dataset_file_name']
        else:
            self.use_embed_offset = False
            self.dataset_file_name = None

        ### baseline config
        # self.generator_baseline = 'Gnp'
        self.generator_baseline = data['generator_baseline']

        # self.metric_baseline = 'general'
        # self.metric_baseline = 'degree'
        self.metric_baseline = data['metric_baseline']


        ### filenames to save intemediate and final outputs
        self.fname = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(self.hidden_size_rnn) + '_'
        self.fname_pred = self.note+'_'+self.graph_type+'_'+str(self.num_layers)+'_'+ str(self.hidden_size_rnn)+'_pred_'
        self.fname_train = self.note+'_'+self.graph_type+'_'+str(self.num_layers)+'_'+ str(self.hidden_size_rnn)+'_train_'
        self.fname_test = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(self.hidden_size_rnn) + '_test_'
        self.fname_baseline = self.graph_save_path + self.graph_type + self.generator_baseline+'_'+self.metric_baseline
