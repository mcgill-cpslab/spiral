"""
For link prediction, the target id list in the input should be a list of tuples, each tuple represent an edge with the
first to be the source and the second to be the destination.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
import datetime
from spiro.core.backend_config import tensorflow
from spiro.dtdg.dataloader import ogb_dataset, supported_ogb_datasets
from spiro.dtdg.models.encoder.implicitTimeEncoder.staticGraphEncoder import GCN, SGCN, GAT, GraphSage
from spiro.dtdg.models.decoder.sequentialDecoders import SelfAttention, PTSA, FTSA, Conv1D
from spiro.dtdg.models.decoder.simpleDecoders import MLP
from spiro.core.commonF import to_tensor
from spiro.automl.model_assembler import assembler
from spiro.automl.tf.prepare_dataset import prepare_edge_task, TARGET, LABEL
from ogb.linkproppred import Evaluator

if __name__ == '__main__':
    #----------------------------------------------------------------
    data_to_test = supported_ogb_datasets()[1]
    this_graph = ogb_dataset(data_to_test)
    n_snapshot = len(this_graph)
    n_nodes = this_graph.dispatcher(n_snapshot -1)[0].observation.num_nodes()
    this_snapshot,_ = this_graph.dispatcher(20)
    in_dim = this_snapshot.num_node_features()
    hidden_dim = 32
    num_GNN_layers = 2
    num_RNN_layers = 2
    output_dim = 10
    activation_f = tf.nn.relu
    encoders = ['gcn', 'sgcn', 'gat', 'sage']
    decoders = ['sa', 'ptsa', 'tsa', 'conv1d']
    epochs = 1000
    eval_t = n_snapshot -1
    prepare_edge_task(this_graph, 2, num_negative=1000, start_t=eval_t)
    eval_snapshot,_ = this_graph.dispatcher(eval_t)
    eval_target_links = this_graph.time_data[TARGET][eval_t]
    edge_to_remove = this_graph.time_data['positive_edges'][eval_t]
    eval_snapshot.observation.remove_edges(edge_to_remove)
    n_pos_eval = len(edge_to_remove)
    
    for g in range(4):
    
        for r in range(4):
            #set up logger
            this_logger = logging.getLogger('citation_predictoin_pipeline')
            this_logger.setLevel(logging.INFO)
            # create file handler which logs even debug messages
            log_path = f"model_{encoders[g]}_{decoders[r]}.log"
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.DEBUG)
            for hdlr in this_logger.handlers[:]:  # remove all old handlers
                this_logger.removeHandler(hdlr)
            this_logger.addHandler(fh)
            this_logger.info("--------------------------------------------------------")
            for trial in range(10):
                this_logger.info("--------------------------------------------------------")
                this_logger.info(f"start trial {trial}")
                if g == 0:
                    gnn = GCN(num_GNN_layers, in_dim, hidden_dim,  activation=activation_f, allow_zero_in_degree=True, dropout=0.2)
                elif g == 1:
                    gnn = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True)
                elif g == 2:
                    gnn = GAT([1], in_dim, hidden_dim,  activation=activation_f,allow_zero_in_degree=True)
                else:
                    gnn = GraphSage('gcn', in_dim, hidden_dim,  activation=activation_f)

                output_decoder = MLP(output_dim, [hidden_dim,20,10,5], activation="sigmoid")
                
                if r == 0:
                    sa = SelfAttention( 3, hidden_dim, [8,16,output_dim], n_nodes, 7, output_decoder)
                elif r == 1:
                    sa = PTSA( 3, hidden_dim, [8,16,output_dim], n_nodes, 7, output_decoder)
                elif r == 2:
                    sa = FTSA( 3, hidden_dim, [output_dim], n_nodes, 7,3,'sum', output_decoder)
                elif r == 3:
                    sa = Conv1D(hidden_dim, [8,16,output_dim], n_nodes, 7, output_decoder)
                else:
                    pass


                this_model = assembler(gnn, sa)
                save_path = f"model_{encoders[g]}_{decoders[r]}_{trial}"
                loss_fn = keras.losses.BinaryCrossentropy()
                loss_list=[]
                all_predictions=[]
                eval_loss = []
                eval_predictions = []
                eval_loss2 = []
                eval_predictions2= []
                lr = 1e-3
                optimizer = keras.optimizers.Adam(learning_rate=lr, epsilon=1e-8)
                for epoch in range(epochs):
                    this_model.decoder.training()
                    this_model.decoder.memory.reset_state()
                    start = 12
                    prepare_edge_task(this_graph, 2, start_t=start)
                    for t in range(start,n_snapshot-1):
                        with tf.GradientTape() as tape:
                            this_snapshot,_ = this_graph.dispatcher(t)
                            target_links = this_graph.time_data[TARGET][t]
                            label = this_graph.time_data[LABEL][t]
                            edge_to_remove = this_graph.time_data['positive_edges'][t]
                            this_snapshot.observation.remove_edges(edge_to_remove)
                            predict = this_model((this_snapshot,target_links))
                            all_predictions.append(tf.squeeze(predict).numpy())
                            loss = loss_fn(tf.squeeze(predict), label)
                            grads = tape.gradient(loss, this_model.trainable_weights)
                            optimizer.apply_gradients(zip(grads, this_model.trainable_weights))
                    test_t = n_snapshot-1
                    this_snapshot,_ = this_graph.dispatcher(test_t)
                    target_links = this_graph.time_data[TARGET][test_t]
                    label = this_graph.time_data[LABEL][test_t]
                    edge_to_remove = this_graph.time_data['positive_edges'][test_t]
                    this_snapshot.observation.remove_edges(edge_to_remove)
                    predict = this_model((this_snapshot,target_links))
                    all_predictions.append(tf.squeeze(predict).numpy())
                    loss = loss_fn(tf.squeeze(predict), label)
                    loss_list.append(loss.numpy())
                    print(loss)

                    mini = min(loss_list)
                    if epoch > 10:
                        if all(loss_list[-20:] > mini):
                            print(mini)
                            break
                        if loss_list[-1] == mini:
                            predict = tf.squeeze(this_model((eval_snapshot,eval_target_links))).numpy()
                            pred_pos = predict[:n_pos_eval]
                            n_nodes_eval = int(pred_pos.shape[0]/2)
                            pred_neg = predict[n_pos_eval:]
                            pred_neg_reshape = []
                            print(f"number of nodes in evaluation: {n_nodes_eval}")
                            for i in range(n_nodes_eval):
                                pred_neg_reshape.append(pred_neg[i*1000:(i+1)*1000])
                                pred_neg_reshape.append(pred_neg[i*1000:(i+1)*1000])
                            pred_neg = np.array(pred_neg_reshape)
                            print(pred_neg.shape)
                            print(pred_pos.shape)
                            evaluator = Evaluator(name =supported_ogb_datasets()[0])
                            input_dict = {"y_pred_pos": pred_pos, "y_pred_neg": pred_neg}
                            result_dict = evaluator.eval(input_dict)
                            print(result_dict.keys())
                            print(result_dict['mrr_list'].mean().item())

                            print(f"save best model for loss {mini}")
                            this_model.save_model(save_path)

                this_logger.info(loss_list)
                this_logger.info(f"best loss {mini}")


