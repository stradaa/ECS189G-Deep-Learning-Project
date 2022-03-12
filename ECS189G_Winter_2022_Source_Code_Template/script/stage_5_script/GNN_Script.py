from stages.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from stages.stage_5_code.Method_GNN import Method_GCN, HistoryDict
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch import Tensor
from tensorflow import keras
from tensorflow.keras import layers
from stages.stage_5_code.Method_GNN import train_step, eval_step, train, plot_history
import torch_geometric.transforms as T


def run_experiment(model, x_train, y_train):
    # Compile the model.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    # Create an early stopping callback.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=50, restore_best_weights=True
    )
    # Fit the model.
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early_stopping],
    )

    return history


def create_ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []

    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    return keras.Sequential(fnn_layers, name=name)


if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # print(torch.cuda.is_available())

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('stage_5_data', '')
    data_obj.dataset_source_folder_path = '../../data/stage_5_data/cora'
    data_obj.dataset_name = 'cora'

    data_test = data_obj.load()  # data_test['edges']

    # print(data_test['graph']['node'])  # length 2708 (idx_map)
    # print(data_test['graph']['edge'])  # length 5429 (edges)
    # print(data_test['graph']['X'])    # length 2708 (features)
    # print(data_test['graph']['y'])    # length 2708 (labels) (7 classes)
    # print(data_test['graph']['utility']['A'])

    # print("LENGTH NODE", data_test['graph']['node'].items())
    # print("LENGTH EDGE", len(data_test['graph']['edge']))
    # print("LENGTH LABELS", len(data_test['graph']['y']))
    # print("LENGTH FEATURES", len(data_test['graph']['X']))
    # print("LENGTH FEATURES", len(data_test['graph']['X'][0]))

    # G = nx.Graph()
    # for idx, i in enumerate(data_test['graph']['X']):
    #     G.add_node(idx, X=i, y=data_test['graph']['y'][idx])
    #
    # G.add_edges_from(data_test['graph']['edge'])
    # nx.draw_spring(G, node_size=5)
    # plt.show()

    SEED = 42
    MAX_EPOCHS = 200
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 5e-4
    EARLY_STOPPING = 10

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Method_GCN(data_test.num_node_features, data_test.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY,
                                 amsgrad=True)
    history = train(model, data_test, optimizer, max_epochs=MAX_EPOCHS, early_stopping=EARLY_STOPPING)

    plt.figure(figsize=(12, 4))
    plot_history(history, "GCN")
