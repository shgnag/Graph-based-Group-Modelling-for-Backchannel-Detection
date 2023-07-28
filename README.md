# Graph-based Group Modelling for Backchannel Detection

This repository conatins the PyTorch implementation of <a href="https://ieeexplore.ieee.org/document/9511820">Graph-based Group Modelling for Backchannel Detection</a> method.

The brief responses given by listeners in group conversations are known as backchannels rendering the task of backchannel detection an essential facet of group interaction analysis. Most of the current backchannel detection studies explore various audio-visual cues for individuals. However, analysing all group members is of utmost importance for backchannel detection, like any group interaction. This study uses a graph neural network to model group interaction through all members' implicit and explicit behaviours. The proposed method achieves the best and second best performance on agreement estimation and backchannel detection tasks, respectively, of the 2022 MultiMediate: Multi-modal Group Behaviour Analysis for Artificial Mediation challenge.

## Dataset
We used the MPIIInteraction dataset for backchannel detection and agreement estimation. Please contact Dr. Philipp Müller, Senior Researcher, DFKI GmbH, Email: philipp.mueller@dfki.de to get the access to the dataset.

## Training
The code can be used to create two different types of graph: Static and dynamic, with different features. Features are extracted prior to training. Please refer to the paper for more details on feature extraction. Update the config.py to set the input paths and use the following command to train a model for graph-based group modelling.

```
python group_modelling_train.py
```

## Citation
If you find the code useful for your research, please consider citing our work:
```
@inproceedings{10.1145/3503161.3551605,
author = {Sharma, Garima and Stefanov, Kalin and Dhall, Abhinav and Cai, Jianfei},
title = {Graph-Based Group Modelling for Backchannel Detection},
year = {2022},
isbn = {9781450392037},
publisher = {Association for Computing Machinery},
url = {https://doi.org/10.1145/3503161.3551605},
doi = {10.1145/3503161.3551605},
booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
pages = {7190–7194}
}
```
## Contact
In case of any questions, please contact garima.sharma1@monash.edu.

## Acknowledgements
This repository uses few modules from <a href="[https://ieeexplore.ieee.org/document/9511820](https://github.com/fuankarion/maas)https://github.com/fuankarion/maas">MAAS</a>.
