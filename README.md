

# Graph Dynamic Earth Net: Spatio-Temporal Graph Benchmark for Satellite Image Time Series

Source code for the paper "**[Graph Dynamic Earth Net: Spatio-Temporal Graph Benchmark for Satellite Image Time Series](https://2023.ieeeigarss.org/index.php)**" by _[Corentin Dufourg](https://www.linkedin.com/in/corentin-dufourg/), [Charlotte Pelletier](https://sites.google.com/site/charpelletier), Stéphane May and [Sébastien Lefèvre](http://people.irisa.fr/Sebastien.Lefevre/)_, at **International Geoscience and Remote Sensing Symposium 2023 (IGARSS 2023)**.

We propose a comparison of five graph neural networks applied to spatio-temporal graphs built from satellite image time series.  
The results highlight the efficiency of graph models in understanding the spatio-temporal context of regions, which might lead to a better classification compared to attribute-based methods.



|       | skip<br>cnnct.  | attnt.                | Nb. of<br>params | Impervious IoU|Agriculture IoU|Forest IoU|Wetlands IoU|Soil IoU|Water IoU | Test<br>mIoU (&uarr;)   | Test<br>Accuracy (&uarr;)   |
| ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RF**    | -      | -                     | -      | 23.07<sub>&plusmn;0.11</sub>    | 0.73<sub>&plusmn;0.03</sub>       | 36.13<sub>&plusmn;0.08</sub> | 0.01<sub>&plusmn;0.04</sub> | 41.04<sub>&plusmn;0.08</sub> | 72.53<sub>&plusmn;0.23</sub> | 28.92<sub>&plusmn;0.06</sub>    | 52.17<sub>&plusmn;0.06</sub>        |
| **MLP**   | -      | -                     | 205k   | 25.21<sub>&plusmn;2.34</sub>    | 0.56<sub>&plusmn;0.25</sub>       | 42.93<sub>&plusmn;1.62</sub> | 0.02<sub>&plusmn;0.04</sub> | 46.61<sub>&plusmn;1.48</sub> | **74.05**<sub>&plusmn;1.60</sub> | 31.57<sub>&plusmn;0.58</sub>    | 57.75<sub>&plusmn;0.89</sub>        |
| **GCN**   | &cross; | &cross;                | 205k   | 2.19<sub>&plusmn;3.76</sub>     | 0.08<sub>&plusmn;0.13</sub>       | 40.65<sub>&plusmn;3.13</sub> | 0.01<sub>&plusmn;0.03</sub> | 37.60<sub>&plusmn;1.87</sub> | 42.02<sub>&plusmn;1.99</sub> | 20.43<sub>&plusmn;0.50</sub>    | 50.69<sub>&plusmn;2.35</sub>        |
| **GSAGE** | &check; | &cross;                | 406k   | <u>30.07</u><sub>&plusmn;3.25</sub> | 1.00<sub>&plusmn;0.80</sub>       | <u>48.50</u><sub>&plusmn;4.22</sub> | 0.17<sub>&plusmn;0.28</sub> | <u>48.67</u><sub>&plusmn;2.05</sub> | 69.90<sub>&plusmn;2.63</sub> | <u>33.06</u><sub>&plusmn;1.46</sub> | <u>60.22</u><sub>&plusmn;2.21</sub>    |
| **GAT**   | &cross; | &check;                | 207k   | 2.19<sub>&plusmn;1.25</sub>     | 0.08<sub>&plusmn;0.12</sub>       | 39.24<sub>&plusmn;2.68</sub> | 0.00<sub>&plusmn;0.00</sub> | 37.05<sub>&plusmn;2.75</sub> | 41.84<sub>&plusmn;1.71</sub> | 20.07<sub>&plusmn;0.84</sub>    | 49.38<sub>&plusmn;1.98</sub>        |
**ResGGCN** | &check; | &check; | 811k | 28.17<sub>&plusmn;2.09</sub> | **2.84**<sub>&plusmn;1.31</sub> | 46.95<sub>&plusmn;1.76</sub> | **0.26**<sub>&plusmn;0.52</sub> | 45.86<sub>&plusmn;1.77</sub> | <u>72.74</u><sub>&plusmn;1.87</sub> | 32.80<sub>&plusmn;0.67</sub> | 59.92<sub>&plusmn;0.89</sub> |
**GT** | &check; | &check; | 2115k | **32.53**<sub>&plusmn;3.09</sub> | <u>1.84</u><sub>&plusmn;1.94</sub> | **50.40**<sub>&plusmn;2.93</sub> | <u>0.23</u><sub>&plusmn;0.52</sub> | **49.74**<sub>&plusmn;3.01</sub> | 71.17<sub>&plusmn;3.16</sub> | **34.31**<sub>&plusmn;1.57</sub> | **61.82**<sub>&plusmn;2.89</sub> |

**Table:** *IoU and Accuracy performance on the test set of attribute-based and GNN models. The results are provided with average and standard deviation on ten random initializations (__best__, <u>second best</u>).*


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


