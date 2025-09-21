import itertools
import ir_datasets
import csv
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_pisa import PisaIndex
from pyterrier.measures import *
import pyterrier_dr
from pyterrier_dr import FlexIndex, TctColBert, TasB
from ir_measures import *
import re
import pandas as pd
import numpy as np
import pyt_splade
import json
import functools as ft

logger = ir_datasets.log.easy()

#----------------------------------------------------------------------
### Get neighbors pyterrier transformer
#----------------------------------------------------------------------

class getn(pt.Transformer):
    def __init__(self, l):
        self.l = l
        with open('graph.json') as json_file:
            self.graph = json.load(json_file)

    def transform(self, inp):
        assert 'docno' in inp.columns and 'qid' in inp.columns
        reslist = []
        for t in tlist:
            for b in blist:
                res = {'qid': [], 'docno': [], 'score': []}
                it = iter(inp.groupby('qid'))
                it = logger.pbar(it)
                for qid, df in it:
                    docnos1 = df['docno'].values
                    # GRAPH !!!
                    #print(int(self.l*t))
                    docnos2 = [self.graph[doc] for doc in docnos1[:int(self.l*t)] if doc in self.graph.keys()]
                    #print(docnos2)
                    if docnos2:
                        unique_elements, frequency = np.unique(np.concatenate(docnos2), return_counts=True)
                        sorted_indexes = np.argsort(frequency)[::-1]
                        docnos2 = unique_elements[sorted_indexes]
                        docnos2 = [str(doc) for doc in docnos2]
                        #print(docnos2)
                        #print(docnos1)
                    docnos3 = list(set(docnos2) - set(docnos1[:self.l - int(self.l*b)]))[:int(self.l*b)]

                    ext_docnos = [docnos1[:self.l - len(docnos3)]]
                    ext_docnos.append(docnos3)
                    # print(len(docnos1))
                    # print(len(docnos2))
                    # print(len(ext_docnos[0]))
                    # print(len(docnos3))
                    # print(ext_docnos)
                    # print(docnos3)
                    final_docids = np.unique(np.concatenate(ext_docnos))

                    res['qid'].extend(itertools.repeat(qid, len(final_docids)))
                    res['docno'].append(final_docids)
                    res['score'].extend(itertools.repeat(1, len(final_docids)))
                    #print(len(final_docids))
                    #print(self.l)
                    #print(len(df['score'].values[:self.l]))
                res['docno'] = np.concatenate(res['docno'])
                #print(len(res['docno']))
                #print(len(res['score']))
                #print(len(res['qid']))
                res = pd.DataFrame(res)
                res = pt.model.add_ranks(res)
                reslist.append(res)

        return reslist

#----------------------------------------------------------------------
### Hyper-parameters
#----------------------------------------------------------------------

qset = 'small'
tlist = [0.01, 0.02, 0.03, 0.04]
blist = [0.1, 0.2, 0.3, 0.4]

#----------------------------------------------------------------------
### Load query
#----------------------------------------------------------------------

df_examples = pd.read_parquet('esci-1/shopping_queries_dataset_examples.parquet')
print(df_examples.head())
df = df_examples[df_examples[qset + "_version"] == 1]
df = df[df["product_locale"] == 'us']
df_train = df[df["split"] == "train"]
df_test = df[df["split"] == "test"]
# print(df_test.head())
# print((df_train['query_id'].nunique()))
# print((df_test['query_id'].nunique()))
query_df = df_test.filter(['query','query_id'], axis=1)
query_df.rename(columns = {'query_id':'qid'}, inplace = True)
query_df = query_df.drop_duplicates()
# print(query_df.head())
# print(len(query_df))

for index, row in query_df.iterrows():
    query_df.at[index, 'query'] = re.sub(r'[^\w\s]', '', row['query'])
    # print(row['query'])
print(query_df.head())

#----------------------------------------------------------------------
### Load qrels
#----------------------------------------------------------------------

qrel_df = df_test.filter(['query_id', 'product_id', 'esci_label'], axis=1)
qrel_df.rename(columns = {'query_id':'qid', 'esci_label': 'label'}, inplace = True)
qrel_df['label'] = qrel_df['label'].map({'E': 3, 'S': 2, 'C': 1, 'I': 0})
print(qrel_df.head())
#print(len(qrel_df))

#----------------------------------------------------------------------
### Load products to assign new docnos
#----------------------------------------------------------------------

df_products = pd.read_parquet('esci-1/shopping_queries_dataset_products.parquet')
df_products = df_products[df_products['product_locale']=='us']
df_products['new_id'] = np.arange(1, len(df_products) + 1)
print((df_products.head()))
new_qrel_df = pd.merge(df_products, qrel_df, on='product_id', how='inner')
new_qrel_df.rename(columns = {'new_id':'docno'}, inplace = True)
print(new_qrel_df.head())
print(len(qrel_df))
print(len(new_qrel_df))

#----------------------------------------------------------------------
### Verify string datatypes
#----------------------------------------------------------------------

new_qrel_df['docno'] = new_qrel_df['docno'].astype(str)
new_qrel_df['qid'] = new_qrel_df['qid'].astype(str)
query_df['qid'] = query_df['qid'].astype(str)
query_df['query'] = query_df['query'].astype(str)
# print(qrels)
print(query_df)
print(len(query_df))

#----------------------------------------------------------------------
### Experiment
#----------------------------------------------------------------------

for base in ['bm25', 'tasb', 'hnp', 'splade']:
    baseres = pd.read_pickle('retrieval_results/' + base + '_' + qset + '.p')

    namelist = [base]
    for t in tlist:
        for b in blist:
            namelist.append(base + '-' + str(int(t*100)) + '-' + str(int(b*100)))

    resdfs = []
    for l in [500, 1000, 1500, 2000]:
        piperes = getn(l).transform(baseres)
        print(piperes)
        reslist = [baseres]
        reslist.extend(piperes)

        met = pt.Experiment(
            reslist,
            query_df,
            new_qrel_df,
            eval_metrics=[R(rel=1)@l],
            names=namelist,
            baseline=0
        )
        resdfs.append(met)
        print(met)


    df_final = ft.reduce(lambda left, right: pd.merge(left, right, on='name'), resdfs)
    print(base)
    print(df_final)
    df_final.to_csv('metric_results/' +base+'-44-' + qset + '.csv')


