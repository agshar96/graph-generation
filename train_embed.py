import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.decomposition import PCA
import logging
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from time import gmtime, strftime
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from random import shuffle
import pickle
from tensorboard_logger import configure, log_value
import scipy.misc
import time as tm

from utils import *
from model import *
from data import *
from args import Args
from args_from_json import Args_from_json
import create_graphs


def train_mlp_embed_epoch(epoch, args, rnn, output, output_embed, data_loader,
                            optimizer_rnn, optimizer_output, optimizer_output_embed,
                            scheduler_rnn, scheduler_output, scheduler_output_embed):
    rnn.train()
    output.train()
    output_embed.train()
    loss_sum_adj = 0
    loss_sum_embed = 0
    mse_loss = torch.nn.MSELoss()
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        output_embed.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        x_embed_unsorted = data['x_embed'].float()
        y_embed_unsorted = data['y_embed'].float()

        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        x_embed_unsorted = x_embed_unsorted[:, 0:y_len_max, :]
        y_embed_unsorted = y_embed_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        x_embed = torch.index_select(x_embed_unsorted, 0, sort_index)
        y_embed = torch.index_select(y_embed_unsorted, 0, sort_index)

        x_concat = torch.cat((x, x_embed), axis=-1)

        x_concat = Variable(x_concat).cuda()
        y = Variable(y).cuda()
        y_embed = Variable(y_embed).cuda()

        h = rnn(x_concat, pack=True, input_len=y_len)
        y_pred = output(h)
        y_pred_embed = output_embed(h)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        # list(np.array(y_len) + 1) is used here, as the way node embeddings were made, they have one more node
        # Remember here, adj_encoded is (n-1) x (n-1) while node_embedding is n
        y_pred_embed = pack_padded_sequence(y_pred_embed, y_len, batch_first=True)
        y_pred_embed = pad_packed_sequence(y_pred_embed, batch_first=True)[0]
        # y_pred_last = y_pred[-1, :].detach().cpu().numpy()
        # y_last = y[-1, :].detach().cpu().numpy()
        # use cross entropy loss
        loss_adj = binary_cross_entropy_weight(y_pred, y)
        # loss_adj.backward()
        loss_embed = mse_loss(y_pred_embed, y_embed)
        loss = loss_adj + loss_embed
        # loss_embed.backward()
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()
        optimizer_output_embed.step()
        scheduler_output_embed.step()


        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss adj: {:.6f}, train loss embed: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs, loss_adj.item() , loss_embed.item(), args.graph_type, args.num_layers, args.hidden_size_rnn))

        # logging
        log_value('loss_adj'+args.fname, loss_adj.item(), epoch*args.batch_ratio+batch_idx)
        log_value('loss_embed'+args.fname, loss_embed.item(), epoch*args.batch_ratio+batch_idx)

        loss_sum_adj += loss_adj.item()
        loss_sum_embed += loss_embed.item()
    return loss_sum_adj/(batch_idx+1), loss_sum_embed/(batch_idx+1)


def train_rnn_embed_epoch(epoch, args, rnn, output, output_embed, data_loader,
                            optimizer_rnn, optimizer_output, optimizer_output_embed,
                            scheduler_rnn, scheduler_output, scheduler_output_embed):
    rnn.train()
    output.train()
    output_embed.train()
    loss_sum_adj = 0
    loss_sum_embed = 0
    mse_loss = torch.nn.MSELoss()
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        output_embed.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        x_embed_unsorted = data['x_embed'].float()
        y_embed_unsorted = data['y_embed'].float()
        y_len_unsorted = data['len']
        # if args.use_small_graph:
        #     G_pred_list = []
        #     for i in range(x_embed_unsorted.size(0)):
        #         adj_pred = decode_adj(y_unsorted[i].cpu().numpy())
        #         adj_pred = adj_pred[:-1]
        #         embed_pred = y_embed_unsorted[i].cpu().numpy()
        #         G_pred = get_graph_embed(adj_pred, embed_pred) # get a graph from zero-padded adj
        #         G_pred_list.append(G_pred)
        #     fname_graph = 'figures_rnn/GT' + str(epoch)
        #     draw_graph_list_embed(G_pred_list[-5:-1], 2,2,  fname = fname_graph)
            

        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        x_embed_unsorted = x_embed_unsorted[:, 0:y_len_max, :]
        y_embed_unsorted = y_embed_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        x_embed = torch.index_select(x_embed_unsorted, 0, sort_index)
        y_embed = torch.index_select(y_embed_unsorted, 0, sort_index)

        x_concat = torch.cat((x, x_embed), axis=-1)
        
        # input, output for output rnn module
        # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)

        output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1)
        output_y = y_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin)-1,0,-1):
            count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
            output_y_len.extend([min(i,y.size(2))]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
        # pack into variable
        x_concat = Variable(x_concat).cuda()
        y = Variable(y).cuda()
        y_embed = Variable(y_embed).cuda()
        output_x = Variable(output_x).cuda()
        output_y = Variable(output_y).cuda()

        h = rnn(x_concat, pack=True, input_len=y_len)
        y_pred_embed = output_embed(h)

        h = pack_padded_sequence(h,y_len,batch_first=True).data # get packed hidden vector
        # reverse h
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).cuda()
        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(args.num_layers-1, h.size(0), h.size(1))).cuda()
        output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0) # num_layers, batch_size, hidden_size
        y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
        output_y = pad_packed_sequence(output_y,batch_first=True)[0]
        # use cross entropy loss
        y_pred_embed = pack_padded_sequence(y_pred_embed, y_len, batch_first=True)
        y_pred_embed = pad_packed_sequence(y_pred_embed, batch_first=True)[0]
        # y_pred_last = y_pred[-1, :].detach().cpu().numpy()
        # y_last = y[-1, :].detach().cpu().numpy()
        # use cross entropy loss
        loss_adj = binary_cross_entropy_weight(y_pred, output_y)
        # loss_adj.backward()
        loss_embed = mse_loss(y_pred_embed, y_embed)
        loss = loss_adj + loss_embed
        # loss_embed.backward()
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()
        optimizer_output_embed.step()
        scheduler_output_embed.step()


        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss adj: {:.6f}, train loss embed: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs, loss_adj.item() , loss_embed.item(), args.graph_type, args.num_layers, args.hidden_size_rnn))

        # logging
        log_value('loss_adj'+args.fname, loss_adj.item(), epoch*args.batch_ratio+batch_idx)
        log_value('loss_embed'+args.fname, loss_embed.item(), epoch*args.batch_ratio+batch_idx)

        loss_sum_adj += loss_adj.item()
        loss_sum_embed += loss_embed.item()
    return loss_sum_adj/(batch_idx+1), loss_sum_embed/(batch_idx+1)


def test_mlp_embed_epoch(epoch, args, rnn, output, output_embed, test_batch_size=16, save_histogram=False,sample_time=1):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()
    output_embed.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    if args.use_small_graph:
        if max_num_node > args.small_node_num:
            max_num_node = args.small_node_num
    y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # normalized prediction score
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
    y_pred_embed = Variable(torch.zeros(test_batch_size, max_num_node, args.node_embedding_size)).cuda()
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node))
    init_embed = torch.from_numpy(np.random.uniform(low=args.node_embed_range[0,:], high=args.node_embed_range[1, :], size=(test_batch_size, args.node_embedding_size)) ).reshape(test_batch_size, 1, args.node_embedding_size).type(torch.float32)
    if args.use_embed_offset:
        prev_init_embed = torch.from_numpy(init_embed.numpy().copy()).cuda()
        init_embed = torch.zeros(test_batch_size, 1, args.node_embedding_size)
    x_step_concat = torch.cat((x_step, init_embed), axis=-1).cuda()
    for i in range(max_num_node):
        h = rnn(x_step_concat)
        y_pred_embed_step = output_embed(h)
        y_pred_step = output(h)
        y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
        y_pred_embed[:, i:i+1, : ] = y_pred_embed_step
        x_step = sample_sigmoid(y_pred_step, sample=True, sample_time=sample_time)
        x_step_concat = torch.cat((x_step, y_pred_embed_step), axis=-1)
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).cuda()
    # y_pred_data = y_pred.data
    y_pred_long_data = y_pred_long.data.long()
    if args.use_embed_offset:
        y_pred_embed_data = y_pred_embed.data + prev_init_embed
    else:
        y_pred_embed_data = y_pred_embed.data

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        if args.use_embed_offset:
            embed_pred = np.concatenate((prev_init_embed[i].cpu().numpy(), y_pred_embed_data[i].cpu().numpy()), axis=0)
        else:
            embed_pred = np.concatenate((init_embed[i].cpu().numpy(), y_pred_embed_data[i].cpu().numpy()), axis=0)
        embed_pred = np.concatenate((init_embed[i].cpu().numpy(), y_pred_embed_data[i].cpu().numpy()), axis=0)
        G_pred = get_graph_embed(adj_pred, embed_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)
    return G_pred_list

def test_rnn_embed_epoch(epoch, args, rnn, output, output_embed, test_batch_size=16):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()
    output_embed.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    if args.use_small_graph:
        if max_num_node > args.small_node_num:
            max_num_node = args.small_node_num
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
    y_pred_embed = Variable(torch.zeros(test_batch_size, max_num_node, args.node_embedding_size)).cuda()
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node))
    init_embed = torch.from_numpy(np.random.uniform(low=args.node_embed_range[0,:], high=args.node_embed_range[1, :], size=(test_batch_size, args.node_embedding_size)) ).reshape(test_batch_size, 1, args.node_embedding_size).type(torch.float32)
    if args.use_embed_offset:
        prev_init_embed = torch.from_numpy(init_embed.numpy().copy()).cuda()
        init_embed = torch.zeros(test_batch_size, 1, args.node_embedding_size)
    x_step_concat = torch.cat((x_step, init_embed), axis=-1).cuda()
    for i in range(max_num_node):
        h = rnn(x_step_concat)
        y_pred_embed_step = output_embed(h)
        y_pred_embed[:, i:i+1, : ] = y_pred_embed_step
        # output.hidden = h.permute(1,0,2)
        hidden_null = Variable(torch.zeros(args.num_layers - 1, h.size(0), h.size(2))).cuda()
        output.hidden = torch.cat((h.permute(1,0,2), hidden_null),
                                  dim=0)  # num_layers, batch_size, hidden_size
        x_step = Variable(torch.zeros(test_batch_size,1,args.max_prev_node)).cuda()
        output_x_step = Variable(torch.ones(test_batch_size,1,1)).cuda()
        for j in range(min(args.max_prev_node,i+1)):
            output_y_pred_step = output(output_x_step)
            output_x_step = sample_sigmoid(output_y_pred_step, sample=True, sample_time=1)
            x_step[:,:,j:j+1] = output_x_step
            output.hidden = Variable(output.hidden.data).cuda()
        y_pred_long[:, i:i + 1, :] = x_step
        x_step_concat = torch.cat((x_step, y_pred_embed_step), axis=-1)
        rnn.hidden = Variable(rnn.hidden.data).cuda()
    
    y_pred_long_data = y_pred_long.data.long()
    if args.use_embed_offset:
        y_pred_embed_data = y_pred_embed.data + prev_init_embed
    else:
        y_pred_embed_data = y_pred_embed.data

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        if args.use_embed_offset:
            embed_pred = np.concatenate((prev_init_embed[i].cpu().numpy(), y_pred_embed_data[i].cpu().numpy()), axis=0)
        else:
            embed_pred = np.concatenate((init_embed[i].cpu().numpy(), y_pred_embed_data[i].cpu().numpy()), axis=0)
        G_pred = get_graph_embed(adj_pred, embed_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)
    return G_pred_list

def train_embed(args, dataset_train, rnn, output, output_embed):
    # check if load existing model
    epoch = 1
    sample_time = 1
    if args.load:
        fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
        rnn.load_state_dict(torch.load(fname))
        fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
        output.load_state_dict(torch.load(fname))

        args.lr = 0.00001
        epoch = args.load_epoch
        print('model loaded!, lr: {}'.format(args.lr))
    else:
        epoch = 1

    # initialize optimizer
    optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=args.lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=args.lr)
    optimizer_output_embed = optim.Adam(list(output_embed.parameters()), lr=args.lr)

    scheduler_rnn = MultiStepLR(optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_output = MultiStepLR(optimizer_output, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_output_embed = MultiStepLR(optimizer_output_embed, milestones=args.milestones, gamma=args.lr_rate)

    # start main loop
    time_all = np.zeros(args.epochs)
    while epoch<=args.epochs:
        time_start = tm.time()
        # train
        if 'GraphRNN_MLP' in args.note:
            train_mlp_embed_epoch(epoch, args, rnn, output, output_embed, dataset_train,
                            optimizer_rnn, optimizer_output, optimizer_output_embed,
                            scheduler_rnn, scheduler_output, scheduler_output_embed)
        elif 'GraphRNN_RNN' in args.note:
            train_rnn_embed_epoch(epoch, args, rnn, output, output_embed, dataset_train,
                            optimizer_rnn, optimizer_output, optimizer_output_embed,
                            scheduler_rnn, scheduler_output, scheduler_output_embed)
        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start
        # test
        print("At epoch: ", epoch)
        if epoch % args.epochs_test == 0 and epoch>=args.epochs_test_start:
            for sample_time in range(1,4):
                G_pred = []
                while len(G_pred)<args.test_total_size:
                    if 'GraphRNN_MLP' in args.note:
                        G_pred_step = test_mlp_embed_epoch(epoch, args, rnn, output, output_embed, test_batch_size=args.test_batch_size,sample_time=sample_time)
                    elif 'GraphRNN_RNN' in args.note:
                        G_pred_step = test_rnn_embed_epoch(epoch, args, rnn, output, output_embed, test_batch_size=args.test_batch_size)
                    G_pred.extend(G_pred_step)
                # save graphs
                fname = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + '.dat'
                save_graph_list(G_pred, fname)
                fname_graph = args.figure_save_path + args.figure_save_name + str(epoch)
                draw_graph_list_embed(G_pred[-3:-1], 1,2,  fname = fname_graph)
                if 'GraphRNN_RNN' in args.note:
                    break
            print('test done, graphs saved')


        # save model checkpoint
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + args.fname + 'lstm_' + str(epoch) + '.dat'
                torch.save(rnn.state_dict(), fname)
                fname = args.model_save_path + args.fname + 'output_' + str(epoch) + '.dat'
                torch.save(output.state_dict(), fname)
        epoch += 1
    np.save(args.timing_save_path+args.fname,time_all)