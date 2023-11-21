# The following file details, the purpose of each config file.

* args\_small\_mlp\_offset.json -- 
    - RNN+MLP, only 100 nodes, the node embeddings are represented as offset from the first node.
    - Used to see if a single layer MLP can decode architecture in a good manner

* args\_small\_offset.json --
    - RNN+RNN, only 100 nodes, offsetted node embeddings
    - The MLP decoder (defined as 'MLP_complex' class) for node embedding is 2 layered. Currently, it isn't controlled by config file. 

* args\_small.json --
    - RNN+RNN, only 100 nodes, absolute value of nodes (no offsets).
    - MLP decoder for node embedding has 2 layers

* args\_subsample\_offset.json --
    - Used for subsampled graph, only 100 nodes, RNN+RNN, offsetted node embeddings.
    - Same as args_small_offset, names of folders to save output is changes mainly.

* args\_subsample\_full.json --
    - Complete nodes, subsampled graph, RNN+RNN
    - File path: "figures\_save\_subsample\_full"

* args\_subsample\_full\_offset.json --
    - Complete nodes, subsampled graph, RNN+RNN, subsampled nodes
    - File path: "figures\_save\_subsample\_full\_offset"

* args\_testrun.json --
    - Use this while debugging.
    - Since, this is for debugging, no pre-configuration is fixed here.

_TO DO - Add a way to mention graph file path to use in case of argoverse_

_TO DO - Add a way to change MLP decoder from config (low priority)_