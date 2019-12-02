import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import networkx as nx
import matplotlib as mpl
import numpy as np
from math import pi
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM
from umap import UMAP
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from datetime import date
from warnings import filterwarnings
import os
import community

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.utils import np_utils
from keras.metrics import categorical_accuracy
from keras.layers import Dropout
import keras.backend as K

filterwarnings('ignore')

# -------   PLOTTING FUNCTIONS -------------------------


def set_plotting_style():
      
    """
    Plotting style parameters, based on the RP group. 
    """    
        
    tw = 1.5

    rc = {'lines.linewidth': 2,
        'axes.labelsize': 18,
        'axes.titlesize': 21,
        'xtick.major' : 16,
        'ytick.major' : 16,
        'xtick.major.width': tw,
        'xtick.minor.width': tw,
        'ytick.major.width': tw,
        'ytick.minor.width': tw,
        'xtick.labelsize': 'large',
        'ytick.labelsize': 'large',
        'font.family': 'sans',
        'weight':'bold',
        'grid.linestyle': ':',
        'grid.linewidth': 1.5,
        'grid.color': '#ffffff',
        'mathtext.fontset': 'stixsans',
        'mathtext.sf': 'fantasy',
        'legend.frameon': True,
        'legend.fontsize': 12, 
       "xtick.direction": "in","ytick.direction": "in"}



    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
    plt.rc('mathtext', fontset='stixsans', sf='sans')
    sns.set_style('ticks', rc=rc)

    #sns.set_palette("colorblind", color_codes=True)
    sns.set_context('notebook', rc=rc)

    rcParams['axes.titlepad'] = 20 


def bokeh_style():

    '''
    Formats bokeh plotting enviroment. Based on the RPgroup PBoC style.
    '''
    theme_json = {'attrs':{'Axis': {
            'axis_label_text_font': 'Helvetica',
            'axis_label_text_font_style': 'normal'
            },
            'Legend': {
                'border_line_width': 1.5,
                'background_fill_alpha': 0.5
            },
            'Text': {
                'text_font_style': 'normal',
               'text_font': 'Helvetica'
            },
            'Title': {
                #'background_fill_color': '#FFEDC0',
                'text_font_style': 'normal',
                'align': 'center',
                'text_font': 'Helvetica',
                'offset': 2,
            }}}

    return theme_json


def get_gene_data(data, gene_name_column, test_gene_list):
    
    """Extract data from specific genes given a larger dataframe.
    
    Inputs
    
    * data: large dataframe from where to filter
    * gene_name_column: column to filter from
    * test_gene_list : a list of genes you want to get
    
    Output
    * dataframe with the genes you want
    """
    
    gene_profiles = pd.DataFrame()

    for gene in data[gene_name_column].values:

        if gene in test_gene_list: 

            df_ = data[(data[gene_name_column] == gene)]

            gene_profiles = pd.concat([gene_profiles, df_])
    
    gene_profiles.drop_duplicates(inplace = True)
    
    return gene_profiles

# ---------PANDAS FUNCTIONS FOR DATA EXPLORATION -------------------------
def count_feature_types(data):
    
    """
    Get the dtype counts for a dataframe's columns. 
    """
    
    df_feature_type = data.dtypes.sort_values().to_frame('feature_type')\
    .groupby(by='feature_type').size().to_frame('count').reset_index()
    
    return df_feature_type


def get_df_missing_columns(data):
    
    '''
    
    Get a dataframe of the missing values in each column with its corresponding dtype.
    
    '''
    
    # Generate a DataFrame with the % of missing values for each column
    df_missing_values = (data.isnull().sum(axis = 0) / len(data) * 100)\
                        .sort_values(ascending = False)\
                        .to_frame('% missing_values').reset_index()
    
    # Generate a DataFrame that indicated the data type for each column
    df_feature_type = data.dtypes.to_frame('feature_type').reset_index()
    
    # Merge frames
    missing_cols_df = pd.merge(df_feature_type, df_missing_values, on = 'index',
                         how = 'inner')

    missing_cols_df.sort_values(['% missing_values', 'feature_type'], inplace = True)
    
    
    return missing_cols_df


def find_constant_features(data):
    
    """
    Get a list of the constant features in a dataframe. 
    """
    const_features = []
    for column in list(data.columns):
        if data[column].unique().size < 2:
            const_features.append(column)
    return const_features


def duplicate_columns(frame):
    '''
    Get a list of the duplicate columns in a pandas dataframe.
    '''
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if np.array_equal(ia, ja):
                    dups.append(cs[i])
                    break
    return dups


def get_duplicate_columns(df):
        
    """
    Returns a list of duplicate columns 
    """
    
    groups = df.columns.to_series().groupby(df.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = df[v].columns
        vs = df[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if np.array_equal(ia, ja):
                    dups.append(cs[i])
                    break
    return dups


def get_df_stats(df):
    
    """
    Wrapper for dataframe stats. 
    
    Output: missing_cols_df, const_feats, dup_cols_list
    """
    missing_cols_df = get_df_missing_columns(df)
    const_features_list = find_constant_features(df)
    dup_cols_list = duplicate_columns(df)

    return missing_cols_df, const_features_list, dup_cols_list


def test_missing_data(df, fname):
    
    """Look for missing entries in a DataFrame."""
    
    assert np.all(df.notnull()), fname + ' contains missing data'



def col_encoding(df, column):
    
    """
    Returns a one hot encoding of a categorical colunmn of a DataFrame.
    
    ------------------------------------------------------------------
    
    inputs~~

    -df:
    -column: name of the column to be one-hot-encoded in string format.
    
    outputs~~
    
    - hot_encoded: one-hot-encoding in matrix format. 
    
    """
    
    le = LabelEncoder()
    
    label_encoded = le.fit_transform(df[column].values)
    
    hot = OneHotEncoder(sparse = False)
    
    hot_encoded = hot.fit_transform(label_encoded.reshape(len(label_encoded), 1))
    
    return hot_encoded


def one_hot_df(df, cat_col_list):
    
    """
    Make one hot encoding on categoric columns.
    
    Returns a dataframe for the categoric columns provided.
    -------------------------
    inputs
    
    - df: original input DataFrame
    - cat_col_list: list of categorical columns to encode.
    
    outputs
    - df_hot: one hot encoded subset of the original DataFrame.
    """

    df_hot = pd.DataFrame()

    for col in cat_col_list:     

        encoded_matrix = col_encoding(df, col)

        df_ = pd.DataFrame(encoded_matrix,
                           columns = [col+ ' ' + str(int(i))\
                                      for i in range(encoded_matrix.shape[1])])

        df_hot = pd.concat([df_hot, df_], axis = 1)
        
    return df_hot


# OTHER FUNCTIONS

def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    
    """
    Wrapper from JakeVDP data analysis handbook
    """
    labels = kmeans.fit_predict(X)

    # plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

    # plot the representation of the KMeans model
    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max()
             for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))

        
@numba.jit(nopython=True)
def draw_bs_sample(data):
    """
    Draw a bootstrap sample from a 1D data set.
    
    Wrapper from J. Bois' BeBi103 course. 
    """
    return np.random.choice(data, size=len(data))


def net_stats(G):
    
    '''Get basic network stats and plots. Specifically degree and clustering coefficient distributions.'''
    
    net_degree_distribution= []

    for i in list(G.degree()):
        net_degree_distribution.append(i[1])
        
    print("Number of nodes in the network: %d" %G.number_of_nodes())
    print("Number of edges in the network: %d" %G.number_of_edges())
    print("Avg node degree: %.2f" %np.mean(list(net_degree_distribution)))
    print('Avg clustering coefficient: %.2f'%nx.cluster.average_clustering(G))
    print('Network density: %.2f'%nx.density(G))

    
    fig, axes = plt.subplots(1,2, figsize = (16,4))

    axes[0].hist(list(net_degree_distribution), bins=20, color = 'lightblue')
    axes[0].set_xlabel("Degree $k$")
    
    #axes[0].set_ylabel("$P(k)$")
    
    axes[1].hist(list(nx.clustering(G).values()), bins= 20, color = 'lightgrey')
    axes[1].set_xlabel("Clustering Coefficient $C$")
    #axes[1].set_ylabel("$P(k)$")
    axes[1].set_xlim([0,1])


def get_network_hubs(ntw):
    
    """
    input: NetworkX ntw
    output:Prints a list of global regulator name and eigenvector centrality score pairs
    """
    
    eigen_cen = nx.eigenvector_centrality(ntw)
    
    hubs = sorted(eigen_cen.items(), key = lambda cc:cc[1], reverse = True)[:10]
    
    return hubs


def get_network_clusters(network_lcc, n_clusters):
    
    """
    input = an empyty list
    
    output = a list with the netoworks clusters
    
    """
    cluster_list = []
    
    for i in range(n_clusters):

        cluster_lcc = [n for n in network_lcc.nodes()\
                       if network_lcc.node[n]['modularity'] == i]

        cluster_list.append(cluster_lcc)

    return cluster_list

def download_and_preprocess_data(org, data_dir = None, variance_ratio = 0.8, 
                                output_path = '~/Downloads/'):
    
    """
    General function to download and preprocess dataset from Colombos. 
    Might have some issues for using with Windows. If you're using windows
    I recommend using the urllib for downloading the dataset. 
    
    Params
    -------
    
    
    data_path (str): path to directory + filename. If none it will download the data
                     from the internet. 
                     
    org (str) : Organism to work with. Available datasets are E. coli (ecoli), 
                B.subtilis (bsubt), P. aeruginosa (paeru), M. tb (mtube), etc. 
                Source: http://colombos.net/cws_data/compendium_data/
                
    variance (float): Fraction of the variance explained to make the PCA denoising. 
    
    Returns
    --------
    
    denoised (pd.DataFrame)
    
    """
    #Check if dataset is in directory
    if data_dir is None:
        
        download_cmd = 'wget http://colombos.net/cws_data/compendium_data/'\
                      + org + '_compendium_data.zip'
        
        unzip_cmd = 'unzip '+org +'_compendium_data.zip'
        
        os.system(download_cmd)
        os.system(unzip_cmd)
        
        df = pd.read_csv('colombos_'+ org + '_exprdata_20151029.txt',
                         sep = '\t', skiprows= np.arange(6))
        
        df.rename(columns = {'Gene name': 'gene name'}, inplace = True)
        
        df['gene name'] = df['gene name'].apply(lambda x: x.lower())
        
    else: 
        
        df = pd.read_csv(data_dir, sep = '\t', skiprows= np.arange(6))
        try : 
            df.rename(columns = {'Gene name': 'gene name'}, inplace = True)
        except:
            pass
    annot = df.iloc[:, :3]
    data = df.iloc[:, 3:]

    preprocess = make_pipeline(SimpleImputer( strategy = 'median'),
                               StandardScaler(), )

    scaled_data = preprocess.fit_transform(data)
    
    # Initialize PCA object
    pca = PCA(variance_ratio, random_state = 42).fit(scaled_data)
    
    # Project to PCA space
    projected = pca.fit_transform(scaled_data)
    
    # Reconstruct the dataset using 80% of the variance of the data 
    reconstructed = pca.inverse_transform(projected)

    # Save into a dataframe
    reconstructed_df = pd.DataFrame(reconstructed, columns = data.columns.to_list())

    # Concatenate with annotation data
    denoised_df = pd.concat([annot, reconstructed_df], axis = 1)
    
    denoised_df['gene name'] = denoised_df['gene name'].apply(lambda x: x.lower())

    # Export dataset 
    denoised_df.to_csv(output_path + 'denoised_' + org + '.csv', index = False)
