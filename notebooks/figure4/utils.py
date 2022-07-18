from scipy.optimize import leastsq
import anndata
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
import pandas as pd
import scanpy as sc
from os.path import join
import numpy as np
import anndata
from matplotlib import rcParams
import pickle
import seaborn as sns

def get_mean_y(ad, pair_k, n_genes_debug=None, key_control='ctrl', method='mean'):
    ka, kb = pair_k.split('+')
    mask_ctrl = (ad.obs['condition'] == key_control)
    
    mask_a = (ad.obs['condition'] == ka + ('+%s' % key_control)) | (ad.obs['condition'] ==  ('%s+' % key_control) + ka)
    mask_b = (ad.obs['condition'] == kb +  ('+%s' % key_control)) | (ad.obs['condition'] ==  ('%s+' % key_control) + kb)
    mask_ab = (ad.obs['condition'] == ka + '+' + kb) | (ad.obs['condition'] == kb + '+' + ka)
    # mask_a_others = (ad.obs['condition'].str.contains(ka)) & (~ad.obs['condition'].str.contains(kb))
    # mask_b_others = (ad.obs['condition'].str.contains(kb)) & (~ad.obs['condition'].str.contains(ka))

    if n_genes_debug is None:
        n_genes_debug = ad.shape[1]
    
    if method == 'mean':
        X = ad[mask_ctrl,:].to_df().to_numpy()[:,:n_genes_debug].mean(axis=0).reshape(n_genes_debug, 1)
        ya = ad[mask_a,:].to_df().to_numpy()[:,:n_genes_debug].mean(axis=0).reshape(n_genes_debug, 1)
        yb = ad[mask_b,:].to_df().to_numpy()[:,:n_genes_debug].mean(axis=0).reshape(n_genes_debug, 1)
        yab = ad[mask_ab,:].to_df().to_numpy()[:,:n_genes_debug].mean(axis=0).reshape(n_genes_debug, 1)
    elif method == 'var':
        X = ad[mask_ctrl,:].to_df().to_numpy()[:,:n_genes_debug].var(axis=0).reshape(n_genes_debug, 1)
        ya = ad[mask_a,:].to_df().to_numpy()[:,:n_genes_debug].var(axis=0).reshape(n_genes_debug, 1)
        yb = ad[mask_b,:].to_df().to_numpy()[:,:n_genes_debug].var(axis=0).reshape(n_genes_debug, 1)
        yab = ad[mask_ab,:].to_df().to_numpy()[:,:n_genes_debug].var(axis=0).reshape(n_genes_debug, 1)

    # print(ya[:5], yb[:5], yab[:5])
    # X.shape, ya.shape, yb.shape, yab.shape
    return (X, ya, yb, yab)

def residuals_one_x_one_y(p, x, y):
    a = p[0]
    intercept = p[1]
    ypred = x * a + intercept
    # print('d', y, ypred)
    return ypred - y

def get_fitted_parms(X, y):    
    X = list(X.flatten())
    y = list(y.flatten())
    
    parms = []
    for i in range(len(X)):
        v = X[i], y[i]
        xi = np.zeros(2) + X[i]
        yi = np.zeros(2) + y[i]
        p = [0, 0]
        # print(i, p, v)
        fitted_parms = leastsq(residuals_one_x_one_y, p, args=(xi, yi))
        
        # print(fitted_parms)
        parms.append(fitted_parms[0])

    return np.array(parms)

def linear_model(x, a, intercept):
    return ((x.T * a) + intercept).T

def plot(X, y, c='red'):
    X, y = X.flatten(), y.flatten()
    plt.plot(X, y, 'o', c=c)
    r2 = r2_score(X, y)
    plt.title('R2=%.2f' % r2)
    return r2
    # plt.legend(['Fit', 'Noisy', 'True'])
    
### Create a function to store values and getting them on demand.
def get_predicted_yab(ad, pair_k, n_genes_debug=None, key_control='ctrl', show=False):
    X, ya, yb, yab = get_mean_y(ad, pair_k, n_genes_debug=n_genes_debug, key_control=key_control)
    
    # print('before fitting')
    if show:
        rcParams['figure.figsize'] = 10, 5
        plt.subplot(1, 3, 1)
        plot(X, ya, c='blue')
        plt.subplot(1, 3, 2)
        plot(X, yb, c='red')
        plt.subplot(1, 3, 3)
        plot(X, yab, c='green')
        plt.close()

    # print('fitting starts')
    parms_a = get_fitted_parms(X, ya)
    parms_b = get_fitted_parms(X, yb)

    a, intercept_a = parms_a[:,0], parms_a[:,1]
    b, intercept_b = parms_b[:,0], parms_b[:,1]
    
    # print('done...')
    # Linear models and plotting functions
    ya_pred = linear_model(X, a, intercept_a)
    yb_pred = linear_model(X, b, intercept_b)

    # predictions of ya, yb and yab based on linear
    yab_est = linear_model(X, a + b, intercept_a + intercept_b)
    return yab_est


def get_means(ori_datadir='/storage/groups/ml01/workspace/mo/for_nacho',
              pred_datadir='/storage/groups/ml01/workspace/carlo.dedonno/cpa-reproducibility/'):
    ori_path = join(ori_datadir, 'Norman2019_prep_new.h5ad')
    ori = sc.read_h5ad(ori_path)

    
    print(os.path.exists(ori_path), ori_path)

    ori_df = ori.to_df()
    ori_df.index = ori.obs.condition
    ori_df = ori_df.groupby(ori_df.index).mean()
    ori_df.columns = ori_df.columns.map(ori.var['gene_symbols'].to_dict())
    ori_df.index = ori_df.index.astype(str)

    # path updated on 2022.05.09
    pred_path = join(pred_datadir, 'pred_real_5613.h5ad') # 'fig4_predicted_adata.h5ad')
    # pred_path = join(pred_datadir, 'fig4_predicted_adata_full.h5ad') # 'fig4_predicted_adata.h5ad')
    print(os.path.exists(pred_path), pred_path)
    pred = sc.read_h5ad(pred_path)
    pred_ad = pred.copy()
    
    # remove controls and/or uncertainty == 0
    pred = pred[(pred.obs['uncertainty_euclidean'] != 0) & (~pred.obs['condition'].str.contains('ctrl')),:]

    # load and easily map predicted values
    # pred_all = predsc.read_h5ad(join(datadir, 'pred_sweep_Norman2019_prep_new_relu_model_seed=16_epoch=80.h5ad'))
    dat = pred.to_df()
    dat['condition'] = pred.obs['condition']
    pred_mean = dat.groupby('condition').mean()
    del dat


    ## Load embeddings
    emb_path = join(pred_datadir, 'figure4_latent_adata.h5ad')
    emb = sc.read_h5ad(emb_path)
    # emb.shape
    print(os.path.exists(emb_path), emb_path)
    
    print('\ndimensions of embedding:', emb.shape)
    
    emb.obs['a'] = emb.obs['condition'].str.split('+').str[0]
    emb.obs['b'] = emb.obs['condition'].str.split('+').str[1]
    emb.obs['value'] = 1

    emb.obs['condition.invariant'] = np.where(emb.obs['a'].astype(str) > emb.obs['b'].astype(str),
                                              emb.obs['a'].astype(str) + '+' + emb.obs['b'].astype(str),
                                              emb.obs['b'].astype(str) + '+' + emb.obs['a'].astype(str))

    ## Prepare a blend of real and simulated cells

    for k in set(emb.obs['condition.invariant']):
        sel = emb[emb.obs['condition.invariant'] == k,:]
        if sel.shape[0] != 2:
            continue
        assert np.array(sel.to_df().T.corr()).min() == 1    
    print('all good. Removing datasets based on repetition is then tractable')

    emb = emb[emb.obs.index.isin(emb.obs.drop_duplicates('condition.invariant').index),:]

    # generate the original data and index it
    pred = pred_mean # .reindex(emb.obs.condition)
    # pred = pred[pred.index.isin(pred_mean.index)]

    pred.columns.equals(ori_df.columns)

    obs_ori = pd.DataFrame(index=ori_df.index)
    obs_ori.index = ori_df.index.astype(str)
    obs_ori['type'] = 'original data'

    obs_pred = pd.DataFrame(index=pred.index)
    obs_pred.index = obs_pred.index.astype(str)
    obs_pred['type'] = 'generated'

    obs_ori.shape, obs_pred.shape

    # pred.obs['condition'].str.contains('control').value_counts()

    ori_df.shape, pred.shape
    pred.columns = ori_df.columns
    means = anndata.AnnData(pd.concat([ori_df, pred]),
                            obs=pd.concat([obs_ori, obs_pred]))

    means.obs['a'] = means.obs.index.str.split('+').str[0]
    means.obs['b'] = means.obs.index.str.split('+').str[1]

    means.obs['condition'] = means.obs.index
    means.obs['type'] = np.where(means.obs['type'] == 'original data', 'measured', 'predicted')

    # assinng uncertainties
    for k in pred_ad.obs:
        # print(k)
        if 'uncertainty' in k:
            means.obs[k] = means.obs['condition'].map(pred_ad.obs.set_index('condition')[k].to_dict())
            means.obs[k] = np.where(pd.isnull(means.obs[k]), 0, means.obs[k])
    
    for gene_name in set(means.obs[['a', 'b']].values.flatten()):
        means.obs['is.%s' % gene_name] = ((means.obs['a'] == gene_name) | (means.obs['b'] == gene_name)).astype(int)

    return means

def plot_heatmap(a, b, means, res=None, n_genes=100, figsize=[7, 3]):
    
    parms = res[((res['a'] == a) & (res['b'] == b)) | ((res['a'] == b) & (res['b'] == a))]
    
    # make sure c1/c2 are always a-b
    # swap coefs
    parms['ctmp']  = np.where(parms['a'] != a, parms['c2'], parms['c1'])
    parms['c2']  = np.where(parms['a'] != a, parms['c1'], parms['c2'])
    parms['c1']  = np.where(parms['a'] != a, parms['ctmp'], parms['c1'])
    # swap labels
    parms['c']  = np.where(parms['a'] != a, parms['b'], parms['a'])
    parms['b']  = np.where(parms['a'] != a, parms['a'], parms['b'])
    parms['a']  = np.where(parms['a'] != a, parms['c'], parms['a'])
    c1, c2, lin_model_fit, dominance, magnitude = parms[['c1', 'c2', 'lin_model_fit', 'dominance', 'magnitude']].values[0]
    
    title = '%s=%.2f, %s=%.2f' % (a, c1, b, c2)

    # de_genes_a = pickle.load(open('../data/de_genes/%s.pkl' % a, 'rb'))
    # de_genes_b = pickle.load(open('../data/de_genes/%s.pkl' % b, 'rb'))
    pkl_path = '../../data/rf_genes/%s_%s.tsv.gz' % (a, b)
    if not os.path.exists(pkl_path):
        pkl_path = '../../data/rf_genes/%s_%s.tsv.gz' % (b, a)
    assert os.path.exists(pkl_path)

    rf_genes = pd.read_csv(pkl_path, index_col=0)
    # rf_genes = pickle.load(open(pkl_path, 'rb'))
    genes = set(rf_genes.head(n_genes)['index'])
    # print(genes)
    # assert False

    rcParams['figure.dpi'] = 100

    # we'll try using highly variable genes to obtain a sense of genes that could be sub-explore
    # sc.pp.highly_variable_genes(sel)
    # print(sel.var['highly_variable'].value_counts())
    cond_ctrl = (means.obs.index == 'ctrl')
    cond_ab = (means.obs['is.%s' % a] == True) & (means.obs['is.%s' % b] == True)
    cond_a = (means.obs['is.%s' % a] == True) & (means.obs['is.ctrl'] == True)
    cond_b = (means.obs['is.%s' % b] == True) & (means.obs['is.ctrl'] == True)

    sel_ctrl = means[cond_ctrl,:].to_df().mean()
    sel_ab = means[cond_ab,:].to_df().mean()
    sel_a = means[cond_a,:].to_df().mean()
    sel_b = means[cond_b,:].to_df().mean()

    annot_ctrl =  means[cond_ctrl,:].obs['type'].head(1)
    annot_ab =  means[cond_ab,:].obs['type'].head(1)
    annot_a =  means[cond_a,:].obs['type'].head(1)
    annot_b =  means[cond_b,:].obs['type'].head(1)
    annot_values = np.array([annot_ab.values, annot_a.values, annot_b.values]).flatten() # , annot_ctrl.values]).flatten()

    # sel = sel[:,sel.var.index.isin(genes_a)]

    # print(sel_ab.shape, sel_a.shape, sel_b.shape, sel_ctrl.shape)
    
    sel = pd.concat([sel_ab, sel_a, sel_b, sel_ctrl], axis=1).T
    sel.index = ['%s+%s' % (a, b), '%s+ctrl' % a, '%s+ctrl' % b, 'ctrl']

    # print('rf genes', genes)
    # print(sel.var_names)
    sel = sel.iloc[:,sel.columns.isin(genes)]


    for col in sel.columns:
        col_zscore = col
        
        mu = sel[sel.index == 'ctrl'][col].values[0]
        sel[col_zscore] = (sel[col] - mu)/sel[col].std(ddof=0)
        
        # sel[col_zscore] = (sel[col] - sel[col].mean())/sel[col].std(ddof=0)

    sel = sel[sel.index != 'ctrl']

    # print(sel.shape)
    # print(sel.head())
    vmax = sel.abs().max().max()
    vmax = int(vmax) + .5
    vmin = -vmax

    # print(vmin, vmax)


    annot = pd.DataFrame(index=sel.index)
    
    print(annot_values)
    annot['is.predicted'] = np.where(annot_values == 'generated', 'black', 'white')

    import matplotlib.pyplot as plt

    sel.index = sel.index.str.replace('ctrl\+', '').str.replace('\+ctrl', '')
    annot.index = annot.index.str.replace('ctrl\+', '').str.replace('\+ctrl', '')
    
    cg = sns.clustermap(sel, cmap='RdBu_r', vmin=vmin, vmax=vmax, row_cluster=True,
                        yticklabels=True, xticklabels=False, figsize=figsize, row_colors=annot,
                        dendrogram_ratio=0.1,
                        cbar_kws={'label': 'Expression (Z-score)'})
    cg.fig.suptitle(title, fontsize=8)
    plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize=7);
    plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize=8);

def get_zscores_features(a, b, means, rf_genes, n_genes=200):
    
    genes = set(rf_genes.head(n_genes).index)
    cond_ctrl = (means.obs.index == 'ctrl')
    cond_ab = (means.obs['is.%s' % a] == True) & (means.obs['is.%s' % b] == True)
    cond_a = (means.obs['is.%s' % a] == True) & (means.obs['is.ctrl'] == True)
    cond_b = (means.obs['is.%s' % b] == True) & (means.obs['is.ctrl'] == True)

    sel_ctrl = means[cond_ctrl,:].to_df().mean()
    sel_ab = means[cond_ab,:].to_df().mean()
    sel_a = means[cond_a,:].to_df().mean()
    sel_b = means[cond_b,:].to_df().mean()

    annot_ctrl =  means[cond_ctrl,:].obs['type'].head(1)
    annot_ab =  means[cond_ab,:].obs['type'].head(1)
    annot_a =  means[cond_a,:].obs['type'].head(1)
    annot_b =  means[cond_b,:].obs['type'].head(1)
    annot_values = np.array([annot_ab.values, annot_a.values, annot_b.values]).flatten() # , annot_ctrl.values]).flatten()

    # sel = sel[:,sel.var.index.isin(genes_a)]

    sel = pd.concat([sel_ab, sel_a, sel_b, sel_ctrl], axis=1).T
    sel.index = ['%s+%s' % (a, b), '%s+ctrl' % a, '%s+ctrl' % b, 'ctrl']

    sel = sel.iloc[:,sel.columns.isin(genes)]


    for col in sel.columns:
        col_zscore = col
        
        mu = sel[sel.index == 'ctrl'][col].values[0]
        sel[col_zscore] = (sel[col] - mu)/sel[col].std(ddof=0)
        
        # sel[col_zscore] = (sel[col] - sel[col].mean())/sel[col].std(ddof=0)

    sel = sel[sel.index != 'ctrl']
    
    return sel

