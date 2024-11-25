This repository contains the implementation of FUGCF.

## Installation
To set up the environment for running FUGCF, follow these steps:

1. Install the required dependencies:
``` 
pip install -r requirements.txt
```

2. Running FUGCF on the datasets of MovieLens-1M
```
bash run.sh
```
3. After successful running, you will receive the following log results, saved in "nohup.out.ml1m":

...
   | end of step    0 | time =  5.84 | HR@10 = 35.9392 | HR@20 = 38.1963 | NDCG@10 = 37.1241 | NDCG@20 = 36.6348 | PSP@10 = 4.9157 | PSP@20 = 5.4427 | RECALL@10 = 21.8311 | RECALL@20 = 31.6985 | PRECISION@10 = 29.7152 | PRECISION@20 = 23.2376 | EUF@10 = 0.1629 | EUF@20 = 0.0821 | AUC = 0.9457 (TEST)
