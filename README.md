# The 2nd YouTube-8M Video Understanding Challenge

The 18th Place Solution for [The 2nd YouTube-8M Video Understanding Challenge](https://www.kaggle.com/c/youtube8m-2018)

## Challenge Description

Similar to the competition last year, Youtube is challenging us to classify 8 Million videos (over 2 TB data). However, this year, participants are required to build a compact model. Specifically, your inference model size should not exceed 1GB. In addition, the rule states that you __must__ upload our models; otherwise, you will be removed from the leaderboard. This indicates that there will be __no more blending of output files__ in this challenge.

## Usage

To train the model:

```
python train.py --frame_features --model=NetVLADModelLF --feature_names='rgb,audio' --feature_sizes='1024,128' --train_data_pattern=input/train*.tfrecord --train_dir=gatednetvladLF-200k-900-80-0002-300iter-norelu-basic-gatedmo5e --start_new_model--batch_size=80 --base_learning_rate=0.0002 --netvlad_cluster_size=200 --netvlad_hidden_size=900 --moe_l2=1e-6 --iterations=300 --learning_rate_decay=0.8 --netvlad_relu=False --gating=True --moe_prob_gating=True --moe_num_mixtures=5 --max_step=245000
```

To evaluate the model:

```
python eval.py --eval_data_pattern=input/validate*.tfrecord  --train_dir=gatednetvladLF-200k-900-80-0002-300iter-norelu-basic-gatedmo5e --frame_features --model=NetVLADModelLF --feature_names='rgb,audio' --feature_sizes='1024,128' --base_learning_rate=0.0002 --netvlad_cluster_size=200 --netvlad_hidden_size=900 --moe_l2=1e-6 --iterations=300 --learning_rate_decay=0.8 --netvlad_relu=False --gating=True --moe_prob_gating=True --moe_num_mixtures=5

```

Generate prediction file and compressed model by doing inference:

```
python inference.py --train_dir=gatednetvladLF-200k-900-80-0002-300iter-norelu-basic-gatedmo5e --output_file=gatednetvladLF-200k-900-80-0002-300iter-norelu-basic-gatedmo5e.csv ----output_model_tgz=gatednetvladLF-200k-900-80-0002-300iter-norelu-basic-gatedmo5e.tgz --input_data_pattern=input/test*.tfrecord 
```
