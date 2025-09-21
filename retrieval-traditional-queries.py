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

#----------------------------------------------------------------------
### Load models and index
#----------------------------------------------------------------------

bm25indexname = './final-esci.lexical'
bm25index = pt.IndexFactory.of(bm25indexname)
bm25 = pt.BatchRetrieve(bm25index, wmodel="BM25", num_results=2000).parallel(12)

model = TasB.dot()
idx = FlexIndex('final-esci.tasb.flex')
tasbpipeline = model.query_encoder() >> idx.np_retriever(num_results=2000)

model = TctColBert('castorini/tct_colbert-v2-hnp-msmarco')
idx = FlexIndex('final-esci.hnp.flex')
hnppipeline = model.query_encoder() >> idx.np_retriever(num_results=2000)

#rrpipeline = bm25 >> model.query_encoder() >> idx.scorer()

factory = pyt_splade.SpladeFactory()
query_encoder = factory.query()
inmemindex = pt.IndexFactory.of("./esci.splade", memory=False)
sppipe = query_encoder >> pt.BatchRetrieve(inmemindex, wmodel='Tf', num_results=2000, verbose=True).parallel(12)
#sppipe.parallel(4)

#----------------------------------------------------------------------
### Query, qrel data function
#----------------------------------------------------------------------

def get_query_df(qset):

    # ----------------------------------------------------------------------
    ### Load query data
    # ----------------------------------------------------------------------

    df_examples = pd.read_parquet('esci-1/shopping_queries_dataset_examples.parquet')
    #print(df_examples.head())

    df = df_examples[df_examples[qset + "_version"] == 1]
    df = df[df["product_locale"] == 'us']
    df_train = df[df["split"] == "train"]
    df_test = df[df["split"] == "test"]
    #print(df_test.head())
    # print((df_train['query_id'].nunique()))
    # print((df_test['query_id'].nunique()))
    query_df = df_test.filter(['query','query_id'], axis=1)
    query_df.rename(columns = {'query_id':'qid'}, inplace = True)
    query_df = query_df.drop_duplicates()
    #print(query_df.head())
    #print(len(query_df))

    for index, row in query_df.iterrows():
        query_df.at[index, 'query'] = re.sub(r'[^\w\s]', '', row['query'])
    print(query_df.head())

    # ----------------------------------------------------------------------
    ### Load qrel data
    # ----------------------------------------------------------------------

    qrel_df = df_test.filter(['query_id', 'product_id', 'esci_label'], axis=1)
    qrel_df.rename(columns = {'query_id':'qid', 'esci_label': 'label'}, inplace = True)
    qrel_df['label'] = qrel_df['label'].map({'E': 3, 'S': 2, 'C': 1, 'I': 0})
    #print(qrel_df.head())
    #print(len(qrel_df))

    # ----------------------------------------------------------------------
    ### Assign new docno in qrel data
    # ----------------------------------------------------------------------

    df_products = pd.read_parquet('esci-1/shopping_queries_dataset_products.parquet')
    df_products = df_products[df_products['product_locale']=='us']
    df_products['new_id'] = np.arange(1, len(df_products) + 1)
    print((df_products.head()))
    new_qrel_df = pd.merge(df_products, qrel_df, on='product_id', how='inner')
    new_qrel_df.rename(columns = {'new_id':'docno'}, inplace = True)
    #print(new_qrel_df.head())
    #print(len(qrel_df))
    #print(len(new_qrel_df))

    # ----------------------------------------------------------------------
    ### Verify dtypes in new qrel df and query df
    # ----------------------------------------------------------------------

    new_qrel_df['docno'] = new_qrel_df['docno'].astype(str)
    new_qrel_df['qid'] = new_qrel_df['qid'].astype(str)
    query_df['qid'] = query_df['qid'].astype(str)
    query_df['query'] = query_df['query'].astype(str)

    #print(query_df)
    #print(len(query_df))

    return query_df, new_qrel_df


#----------------------------------------------------------------------
### Experiment baselines and save retrieval results
#----------------------------------------------------------------------

for qset in ['small', 'large']:
    query_df, qrel_df = get_query_df(qset)

    bm25res = bm25.transform(query_df)
    bm25res.to_pickle('retrieval_results/bm25_' + qset + '.p')
    # bm25res = pd.read_pickle('retrieval_results/bm25_' + qset + '.p')

    tasbres = tasbpipeline.transform(query_df)
    tasbres.to_pickle('retrieval_results/tasb_' + qset + '.p')
    # tasbres = pd.read_pickle('retrieval_results/tasb_' + qset + '.p')

    hnpres = hnppipeline.transform(query_df)
    hnpres.to_pickle('retrieval_results/hnp_' + qset + '.p')
    # hnpres = pd.read_pickle('retrieval_results/hnp_' + qset + '.p')

    resgen = sppipe.transform_gen(query_df, batch_size=1024)
    resdfs = []
    for outt in resgen:
        resdfs.append(outt)
    spres = pd.concat(resdfs)
    spres.to_pickle('retrieval_results/splade_' + qset + '.p')
    # spres = pd.read_pickle('retrieval_results/splade_' + qset + '.p')


    met = pt.Experiment(
        [bm25res, tasbres, hnpres, spres],
        query_df,
        qrel_df,
        eval_metrics=[R@500, R@1000, R@1500, R@2000],
        names=['BM25', 'TASB', 'HNP', 'SP++']
    )

    print(met)
