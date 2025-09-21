import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_pisa import PisaIndex
from pyterrier.measures import *
import pyterrier_dr
from pyterrier_dr import FlexIndex, TctColBert, TasB
import pyt_splade
import pandas as pd
import numpy as np
from numpy import save, load
import ir_datasets
import json

#----------------------------------------------------------------------
### Load train data
#----------------------------------------------------------------------

df_examples = pd.read_parquet('esci-1/shopping_queries_dataset_examples.parquet')
print(df_examples.head())
df = df_examples[df_examples["large_version"] == 1]
df = df[df["product_locale"] == 'us']
df_train = df[df["split"] == "train"]
df_test = df[df["split"] == "test"]

df_products = pd.read_parquet('esci-1/shopping_queries_dataset_products.parquet')
df_products = df_products[df_products['product_locale']=='us']
df_products['product_bullet_point'] = df_products['product_bullet_point'].str.replace('\n','. ')
df_products['new_id'] = np.arange(1, len(df_products) + 1)
print((df_products.head()))
new_df = pd.merge(df_products, df_train, on='product_id', how='inner')
new_df.rename(columns = {'new_id':'docno'}, inplace = True)

new_df = new_df.filter(['query_id','docno', 'esci_label'], axis=1)
print(new_df['esci_label'].unique())
new_df = new_df[new_df['esci_label'] != 'I']
print(new_df.head())
print(new_df['esci_label'].unique())
#print((df_train['product_id'].nunique()))
#print((new_df['product_id'].nunique()))
print(len(new_df))
dictt = {}
ctr = 0
logger = ir_datasets.log.easy()

it = iter(new_df.groupby('query_id'))
it = logger.pbar(it)

#----------------------------------------------------------------------
### Define weight matrix
#----------------------------------------------------------------------

fd = {'E': {'E': 3, 'S': 2, 'C': 1}, 'S':{'E': 2, 'S': 2, 'C': 1}, 'C':{'E': 1, 'S': 1, 'C': 1}}

#----------------------------------------------------------------------
### Build graph
#----------------------------------------------------------------------

ctr = 0
for qid, df in it:
    docs = df['docno'].tolist()
    labels = df['esci_label'].tolist()
    #print(docs)
    #print(labels)
    for i in range(len(docs)):
        for j in range(len(docs)):
            if i == j:
                continue
            w = fd[labels[i]][labels[j]]
            if docs[i] not in dictt.keys():
                dictt[docs[i]] = []
                for k in range(w):
                    dictt[docs[i]].append(docs[j])
            else:
                for k in range(w):
                    dictt[docs[i]].append(docs[j])

    # ctr += 1
    # if ctr %1000 == 0:
    #     print(ctr)
#print(dictt)

#----------------------------------------------------------------------
### Save graph
#----------------------------------------------------------------------

with open("graph.json", "w") as outfile:
    json.dump(dictt, outfile)
