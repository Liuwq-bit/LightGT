# LightGT: A Light Graph Transformer for Multimedia Recommendation
This is our Pytorch implementation for the [LightGT](https://dl.acm.org/doi/10.1145/3539618.3591716):  
> Yinwei Wei, Wenqi Liu, Fan Liu, Xiang Wang, Liqiang Nie and Tat-Seng Chua (2023). LightGT: A Light Graph Transformer for Multimedia Recommendation. In ACM SIGIR`23, Taipei, July. 23-27, 2023

<img src="https://github.com/Liuwq-bit/LightGT/blob/master/image/figure1.png" width="50%" height="50%"><img src="https://github.com/Liuwq-bit/LightGT/blob/master/image/figure2.png" width="50%" height="50%">

## Environment Requirement
The code has been tested running under Python 3.8.15. The required packages are as follows:
- Pytorch == 1.7.0
- numpy == 1.23.4

## Example to Run the Codes
The instruction of commands has been clearly stated in the codes.

- Movielens dataset  
  `python main.py --l_r=1e-2 --weight_decay=1e-2 --src_len=50 --score_weight=0.05 --nhead=1 --transformer_layers=4 --batch_size=2048 --lightgcn_layers=4 --dataset=movielens`  
- Tiktok dataset  
  `python main.py --l_r=1e-2 --weight_decay=1e-2 --src_len=50 --score_weight=0.05 --nhead=1 --transformer_layers=4 --batch_size=2048 --lightgcn_layers=4 --dataset=tiktok`

- Kwai dataset  
```python main.py --l_r=1e-2 --weight_decay=1e-2 --src_len=50 --score_weight=0.05 --nhead=1 --transformer_layers=4 --batch_size=2048 --lightgcn_layers=4 --dataset=kwai```

## Dataset
You can find the full version of recommendation datasets via [Kwai](https://www.kuaishou.com/activity/uimc), [Tiktok](http://ai-lab-challenge.bytedance.com/tce/vc/), and [Movielens](https://grouplens.org/datasets/movielens/).
Since the copyright of datasets, we cannot release them directly. 

||#Interactions|#Users|#Items|Visual|Acoustic|Textual|
|:-|:-|:-|:-|:-|:-|:-|
|Movielens|1,239,508|55,485|5,986|2,048|128|100|
|Tiktok|726,065|36,656|76,085|128|128|128|
|Kwai|1,664,305|22,611|329,510|2,048|-|100|

It is worth noting that [MMGCN](https://github.com/weiyinwei/MMGCN) provides corresponding toy datasets that can be used for research.

-`train.npy`
   Train file. Each line is a user with her/his positive interactions with items: (userID and micro-video ID)  
-`val.npy`
   Validation file. Each line is a user several positive interactions with items: (userID and micro-video ID)  
-`test.npy`
   Test file. Each line is a user with several positive interactions with items: (userID and micro-video ID)  

## Citation
If you want to use our codes and datasets in your research, please cite:

``` 
@inproceedings{wei2023lightgt,
  title      = {Lightgt: A light graph transformer for multimedia recommendation},
  author     = {Wei, Yinwei and
                Liu, Wenqi and
                Liu, Fan and
                Wang, Xiang and
                Nie, Liqiang and
                Chua, Tat-Seng},
  booktitle  = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages      = {1508--1517},
  year       = {2023}
}
```
