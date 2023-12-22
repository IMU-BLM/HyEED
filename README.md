# HyEED: Embedding Learning of Knowledge Graphs with Entity Description in Hyperbolic Space

This is the code of paper **HyEED: Embedding Learning of Knowledge Graphs with Entity Description in Hyperbolic Space.** Liming Bao, Yan Wang, Jiantao Zhou, Xiaoyu Song.

A Knowledge graph aim to use graph structures to model, identify and infer complex relations and domain knowledge among things. It serves as a crucial cornerstone for achieving cognitive intelligence. Typically, knowledge graphs exhibit hierarchical structures, and many studies have demonstrated that hyperbolic embedding methods exhibit strong learning capabilities in learning data with hierarchical structures. It can effectively capture hierarchical structures in a high-fidelity and simplified form. Nevertheless, current hyperbolic embedding models only focus on learning knowledge triples of relations between entities, making the embedding space relatively one-sided. They ignore the entity description, category and other additional information containing rich features, which can make graph embedding learn richer semantic information. Therefore, to address this limitation, we propose a hyperbolic knowledge graph embedding with entity description (\textbf{HyEED}), which can further enrich and compensate for graph structure information, thus achieving better embedding effects. Specifically, we use hyperbolic space-based word embedding model and graph embedding model to encode entity description text information and graph structure information, respectively, and effectively combine them using hyperbolic pooling techniques. Extensive experimental results have unequivocally validated the efficacy and superiority of HyEED.

## Requirments

- python 3.10 - 3.7
- torch 1.13 - 1.8
- numpy 1.24

## Train and Evaluation

- please modify the data set address and output address before running.
- Train and Evaluation 

```python
python main.py --dataset $dataset --datasetdir $datasetdir --outputdir $outputdir --num_iterations --$num_iterations --batch_size $batch_size --lr $learning-rate --max_length $max_length --nneg $negative-samples
```

## Examples

- FB15k-237

```
python main.py --dataset FB15k --datasetdir ./data/ --outputdir ./output/model/ --num_iterations --500 --batch_size 128 --lr 10 --max_length 100 --nneg 50
```

## Acknowledgement

We refer to the code of [Poincaré GloVe](https://github.com/alex-tifrea/poincare_glove), [HyperText](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/HyperText) and [MURP](https://github.com/ibalazevic/multirelational-poincare). Thanks for their contributions.
