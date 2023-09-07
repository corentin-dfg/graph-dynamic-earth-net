

# Graph Dynamic Earth Net: Spatio-Temporal Graph Benchmark for Satellite Image Time Series

Source code for the paper "**[Graph Dynamic Earth Net: Spatio-Temporal Graph Benchmark for Satellite Image Time Series](https://2023.ieeeigarss.org/index.php)**" by _[Corentin Dufourg](https://www.linkedin.com/in/corentin-dufourg/), [Charlotte Pelletier](https://sites.google.com/site/charpelletier), Stéphane May and [Sébastien Lefèvre](http://people.irisa.fr/Sebastien.Lefevre/)_, at **International Geoscience and Remote Sensing Symposium 2023 (IGARSS 2023)**.

We propose a comparison of five graph neural networks applied to spatio-temporal graphs built from satellite image time series.  
The results highlight the efficiency of graph models in understanding the spatio-temporal context of regions, which might lead to a better classification compared to attribute-based methods.



|       | skip<br>cnnct.  | attnt.                | Nb. of<br>params | Impervious IoU|Agriculture IoU|Forest IoU|Wetlands IoU|Soil IoU|Water IoU | Test<br>mIoU (&uarr;)   | Test<br>Accuracy (&uarr;)   |
| ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RF**    | -      | -                     | -      | 22.86<sub>&plusmn;0.38</sub>    | 3.28<sub>&plusmn;0.08</sub>       | 34.73<sub>&plusmn;0.14</sub> | 0.31<sub>&plusmn;0.14</sub> | 41.97<sub>&plusmn;0.33</sub> | 77.28<sub>&plusmn;0.47</sub> | 30.07<sub>&plusmn;0.10</sub>    | 51.95<sub>&plusmn;0.17</sub>        |
| **MLP**   | -      | -                     | 205k   | 19.51<sub>&plusmn;3.11</sub>    | 0.85<sub>&plusmn;0.56</sub>       | 41.86<sub>&plusmn;1.59</sub> | 0.43<sub>&plusmn;0.71</sub> | <ins>48.92</ins><sub>&plusmn;1.60</sub> | 80.75<sub>&plusmn;2.24</sub> | 32.05<sub>&plusmn;0.46</sub>    | 58.33<sub>&plusmn;1.37</sub>        |
| **GCN**   | &cross; | &cross;                | 205k   | 13.94<sub>&plusmn;4.44</sub>     | 1.73<sub>&plusmn;2.22</sub>       | 40.14<sub>&plusmn;1.56</sub> | 0.00<sub>&plusmn;0.00</sub> | 44.75<sub>&plusmn;1.03</sub> | 66.62<sub>&plusmn;2.18</sub> | 27.86<sub>&plusmn;1.23</sub>    | 55.69<sub>&plusmn;1.40</sub>        |
| **GSAGE** | &check; | &cross;                | 406k   | <ins>28.36</ins><sub>&plusmn;4.21</sub> | <ins>4.86</ins><sub>&plusmn;2.38</sub>       | **47.27**<sub>&plusmn;3.04</sub> | 0.86<sub>&plusmn;0.55</sub> | **51.97**<sub>&plusmn;2.15</sub> | <ins>83.46</ins><sub>&plusmn;1.51</sub> | **36.13**<sub>&plusmn;1.33</sub> | **63.01**<sub>&plusmn;2.06</sub>    |
| **GAT**   | &cross; | &check;                | 207k   | 20.96<sub>&plusmn;4.84</sub>     | **9.88**<sub>&plusmn;5.28</sub>       | 41.07<sub>&plusmn;2.82</sub> | <ins>2.11</ins><sub>&plusmn;1.74</sub> | 47.86<sub>&plusmn;2.66</sub> | 76.61<sub>&plusmn;5.15</sub> | 33.08<sub>&plusmn;2.15</sub>    | 58.42<sub>&plusmn;2.32</sub>        |
**ResGGCN** | &check; | &check; | 811k | **30.13**<sub>&plusmn;5.92</sub> | 4.65<sub>&plusmn;3.26</sub> | <ins>43.93</ins><sub>&plusmn;3.62</sub> | **3.28**<sub>&plusmn;1.99</sub> | 48.89<sub>&plusmn;4.56</sub> | **83.93**<sub>&plusmn;1.38</sub> | <ins>35.81</ins><sub>&plusmn;1.91</sub> | <ins>60.19</ins><sub>&plusmn;3.65</sub> |
<!--**GT** | &check; | &check; | 2115k | 3.70<sub>&plusmn;5.93</sub> | 4.71<sub>&plusmn;6.82</sub> | 40.80<sub>&plusmn;16.17</sub> | 0.06<sub>&plusmn;0.18</sub> | 27.32<sub>&plusmn;18.86</sub> | 40.78<sub>&plusmn;25.52</sub> | 19.55<sub>&plusmn;2.16</sub> | 47.23<sub>&plusmn;10.04</sub> |-->

**Table:** *IoU and Accuracy performance on the test set of attribute-based and GNN models. The results are provided with average and standard deviation on ten random initializations (__best__, <u>second best</u>).*


> [!IMPORTANT]  
> :warning: **Disclaimer on IGARSS paper results:** The results provided here override those of the original paper by fixing a problem in graph creation.

<br>

## Overview

* ```train_nn.py``` is the method to train the neural network models.
* ```train_rf.py``` is the method to reproduce the random forest results.
* ```config/``` contains the configuration files to reproduce the paper results.
* ```data/``` contains dataset and spatio-temporal graph classes.
* ```network/``` contains the architectures implemented with PyTorch and PyG.
* ```datasets/``` should contain the data from Dynamic Earth Net and the derived graphs to train and evaluate the models.
* ```helper/``` contains the parser class for command-line options.
* ```utils/``` contains utilitary methods to compute metrics, positional encoding and plot spatio-temporal graph.
* ```requirements.txt``` indicated the required libraries.

<br>

## Usage

```
> python3 train_nn.py --help
usage: train_nn.py [-h] [--results_dir RESULTS_DIR] [--name NAME] [--gpu_ids GPU_IDS] [--checkpoint CHECKPOINT] [--config CONFIG] [--num_workers NUM_WORKERS] [--seed SEED]

options:
  -h, --help            show this help message and exit
  --results_dir RESULTS_DIR, -o RESULTS_DIR
                        models are saved here (default: results_directory)
  --name NAME           name of the experiment. It decides where to store samples and models (default: name_exp)
  --gpu_ids GPU_IDS     gpu ids: e.g. 0 0,1 (default: 0)
  --checkpoint CHECKPOINT
                        Path of a pretrained model to load weights before resuming training. If None random weights are applied (default: None)
  --config CONFIG       Path to the config file. (default: config/mlp.yaml)
  --num_workers NUM_WORKERS
  --seed SEED
```

The model weights, Tensorboard tracks and logs outputs to the ```results_directory``` after the training.
Similar command can be used for Random Forest script.
<br>

## Datasets

To build spatio-temporal graphs from scratch, download the Dynamic Earth Net dataset from [here](https://mediatum.ub.tum.de/1459253?sortfield0=&sortfield1=&show_id=1650201). Then copy the files to ```datasets/den/``` or update the data path on the ```config``` files. Note that for the paper we only keep the first date of each month to stay in a fully supervised context.  

To use the precomputed spatio-temporal graphs from our paper, download [Graph Dynamic Earth Net data in Zenodo]() and move them to the ```datasets/den/processed/``` folder. The data already provided in GitHub are sufficient to reproduce the results, but not to reconstruct the datacube and label the region. For that, you have to recompute the graphs by deleting the ```processed/``` folder and run the script.

The spatio-temporal graphs are structured as follow, derived from the [```HeteroData```](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData) class of PyG:
```
STG(
  img_dir='path/to/original/SITS/folder/',
  label_dir='path/to/the/corresponding/labels/folder/',
  num_nodes,      # Number of nodes in the spatio-temporal graph
  superpixels,    # The correspondance map to link nodes with the region position (Not provided in the precomputed graphs).
                  #   Use the ```reconstruct_to_datacube``` method to reconstruct the datacube from the graph when the superpixels map exists.
  EigVals,        # For node positional encoding
  EigVecs,        # For node positional encoding
  region={
    x,            # Node features
    y,            # Node labels
    pos,          # Node position in a Euclidean space
    node_index    # Node id
  },
  (region, spatial, region)={
    edge_index,   # Nodes from the same date linked by edges 
    edge_attr     # Spatial edge features
  },
  (region, temporal, region)={
    edge_index,   # Nodes from two consecutive dates linked by edges 
    edge_attr     # Temporal edge features
  }
)
```

Note that you can build spatio-temporal graphs and use this code with your own SITS dataset. Some public SITS datasets ready-to-download are listed [here](https://github.com/corentin-dfg/Satellite-Image-Time-Series-Datasets).

### License Information
The data derived from [Dynamic Earth Net](https://openaccess.thecvf.com/content/CVPR2022/papers/Toker_DynamicEarthNet_Daily_Multi-Spectral_Satellite_Dataset_for_Semantic_Change_Segmentation_CVPR_2022_paper.pdf) data cubes are open sourced and ruled by the license [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

<br>

## Reproducibility 

The split used for the data is given in folder ```datasets/den/splits/```. The seeds used for the 10 random initializations are ```0,1,2,3,4,5,6,7,8,9```.

<br>

## Citation

:page_with_curl: Paper in coming
<!--```
@INPROCEEDINGS{dufourg2023graph,
  author={Dufourg, Corentin and Pelletier, Charlotte and May, Stéphane and Lefèvre, Sébastien},
  booktitle={IGARSS 2023 - 2023 IEEE International Geoscience and Remote Sensing Symposium}, 
  title={Graph Dynamic Earth Net: Spatio-Temporal Graph Benchmark for Satellite Image Time Series}, 
  year={2023}
}
```-->


