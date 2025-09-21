# TQE and GRIT 
This repository contains code to reproduce results for paper 'GRIT: Graph-based Recall Improvement for Task-oriented E-commerce Queries' published in Companion Proceedings of the ACM on Web Conference 2025.

If you use this code please cite:
```
@inproceedings{10.1145/3701716.3717859,
author = {Kulkarni, Hrishikesh and Kallumadi, Surya and MacAvaney, Sean and Goharian, Nazli and Frieder, Ophir},
title = {GRIT: Graph-based Recall Improvement for Task-oriented E-commerce Queries},
year = {2025},
isbn = {9798400713316},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3701716.3717859},
doi = {10.1145/3701716.3717859},
abstract = {Many e-commerce search pipelines have four stages, namely: retrieval, filtering, ranking, and personalized-reranking. The retrieval stage must be efficient and yield high recall because relevant products missed in the first stage cannot be considered in later stages. This is challenging for task-oriented queries (queries with actionable intent) where user requirements are contextually intensive and difficult to understand. To foster research in the domain of e-commerce, we created a novel benchmark for Task-oriented Queries (TQE) by using LLM, which operates over the existing ESCI product search dataset. Furthermore, we propose a novel method 'Graph-based Recall Improvement for Task-oriented queries' (GRIT) to address the most crucial first-stage recall improvement needs. GRIT leads to robust and statistically significant improvements over state-of-the-art lexical, dense, and learned-sparse baselines. Our system supports both traditional and task-oriented e-commerce queries, yielding up to 6.3\% recall improvement. In the indexing stage, GRIT first builds a product-product similarity graph using user clicks or manual annotation data. During retrieval, it locates neighbors with higher contextual and action relevance and prioritizes them over the less relevant candidates from the initial retrieval. This leads to a more comprehensive and relevant first-stage result set that improves overall system recall. Overall, GRIT leverages the locality relationships and contextual insights provided by the graph using neighboring nodes to enrich the first-stage retrieval results. We show that the method is not only robust across all introduced parameters, but also works effectively on top of a variety of first-stage retrieval methods.},
booktitle = {Companion Proceedings of the ACM on Web Conference 2025},
pages = {2722–2731},
numpages = {10},
keywords = {e-commerce product search, llm for data generation, product similarity graph, task-oriented queries},
location = {Sydney NSW, Australia},
series = {WWW '25}
}
```

With this paper we release the first Task-oriented Query set for E-commerce (TQE) benchmark as a evaluation forum for e-commerce setting. We also introduce a novel Graph-based Recall Improvement for Task-oriented Queries (GRIT) method which improves first-stage retrieval recall. GRIT leads to statistically significant improvements across SOTA lexical, neural and learned-sparse first-stage retrieval methods. GRIT is also robust across seed size parameter t, product replacement parameter b and handles both traditional and task-oriented queries.

## Benchmark

### TQE.csv
Task-oriented Query set for E-commerce (TQE) benchmark consisting of 22742 English task-oriented queries.

## Code

### create_index.py
Creates BM25, TCT-ColBert-HNP, TAS-B and SPLADE++ indexes on the products in ESCI product data.

### create_graph.py
Creates product-product similarity graph using user-click/train-annotation data.

### retrieval-traditional-queries.py
Evaluates and stores results for BM25, TCT-ColBert-HNP, TAS-B and SPLADE++ for traditional ESCI product search queries.

### retrieval-task-oriented-queries.py
Evaluates and stores results for BM25, TCT-ColBert-HNP, TAS-B and SPLADE++ for task-oriented e-commerce queries from TQE benchmark.

### grit-traditional-queries.py
Evaluates GRIT across seed size parameter t and product replacement parameter b combinations using stored initial retrieval results on traditional queries from retrieval-traditional-queries.py.

### grit-task-oriented-queries.py
Evaluates GRIT across seed size parameter t and product replacement parameter b combinations using stored initial retrieval results on task-oriented queries (TQE benchmark) from retrieval-task-oriented-queries.py.

