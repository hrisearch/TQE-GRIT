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


#----------------------------------------------------------------------
### Data Analysis
#----------------------------------------------------------------------

# df_examples = pd.read_parquet('esci-1/shopping_queries_dataset_examples.parquet')
# print(df_examples.head())
# df = df_examples[df_examples["large_version"] == 1]
# df_train = df_task_1[df["split"] == "train"]
# df_test = df_task_1[df["split"] == "test"]
#
# print((df_train['query_id'].nunique()))
# print((df_test['query_id'].nunique()))

#----------------------------------------------------------------------
### Load products
#----------------------------------------------------------------------

df_products = pd.read_parquet('esci-1/shopping_queries_dataset_products.parquet')
print(df_products.head())
print(len(df_products))

df_products = df_products[df_products['product_locale']=='us']
df_products['product_bullet_point'] = df_products['product_bullet_point'].str.replace('\n','. ')

df_products['new_id'] = np.arange(1, len(df_products) + 1)
df_products['new_id'] = df_products['new_id'].astype(str)

print((df_products.head()))

#----------------------------------------------------------------------
### Create product generator
#----------------------------------------------------------------------

def generate_esci():
    ctr = 0
    for index, row in df_products.iterrows():
        #print(row)
        ctr += 1
        rtext = ''
        for col in ['product_title', 'product_description', 'product_bullet_point', 'product_brand', 'product_color']:
            if row[col] is not None:
                rtext += row[col] + ' '
        #print(row['product_id'])
        # if ctr == 1500:
        #     break
        yield {'docno': row['new_id'], 'text': rtext}


#----------------------------------------------------------------------
### Index products
#----------------------------------------------------------------------

#----------------------------------------------------------------------
### BM25
#----------------------------------------------------------------------

bm25indexname = './final-esci.lexical'
iter_indexer = pt.IterDictIndexer(bm25indexname, meta={'docno': 48, 'text': 8192})
iter_indexer.index(generate_esci())

#----------------------------------------------------------------------
### TCTColBertHNP
#----------------------------------------------------------------------

model = TctColBert('castorini/tct_colbert-v2-hnp-msmarco')
idx = FlexIndex('final-esci.hnp.flex')
pipeline = model.doc_encoder() >> idx
pipeline.index(generate_esci())

idx = FlexIndex('final-esci.hnp.flex')
print(idx.__len__())

#----------------------------------------------------------------------
### TasB
#----------------------------------------------------------------------

model = TasB.dot()
idx = FlexIndex('final-esci.hnp.flex')
pipeline = model.doc_encoder() >> idx
pipeline.index(generate_esci())

idx = FlexIndex('final-esci.hnp.flex')
print(idx.__len__())

#----------------------------------------------------------------------
### SPLADE++
#----------------------------------------------------------------------

indexer = pt.IterDictIndexer('./esci.splade', meta={'docno': 48, 'text': 100000})
factory = pyt_splade.SpladeFactory()
doc_encoder = factory.indexing()

indxr_pipe = (doc_encoder >> pyt_splade.toks2doc() >> indexer)
index_ref = indxr_pipe.index(generate_esci(), batch_size=128)


