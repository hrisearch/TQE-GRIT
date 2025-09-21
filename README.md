# TQE and GRIT 
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

