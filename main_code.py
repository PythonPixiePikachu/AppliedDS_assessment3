import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics as skmet
import sklearn.cluster as cluster


def read_file(path):
    """This function will take path as a parameter
       and reads the csv file and returns two data frames one with 
       country as columns and other witgh year as columns"""
    ds_raw = pd.read_csv(filepath_or_buffer=path, sep=',',
                         encoding='cp1252', skip_blank_lines=True)
    #Deleting rows with null values
    ds_raw = ds_raw.fillna(ds_raw.median())
    pd_countries = pd.DataFrame(ds_raw)
    #calling yearly function inorder to get the years as column
    pd_years = yearly_data(pd_countries)
    out = [pd_countries, pd_years]
    return out


def df_info(df):
    """This function takes data frame as input and returns
       structure of the data frame such as columns,head,tail
       ,transpose,summary"""
    print('Columns of the Data Frame\n')
    print(df.columns)
    print('\n\n')
    print('The top values of Data Frame\n')
    print(df.head())
    print('\n\n')
    print('The bottom values of Data Frame\n')
    print(df.tail())
    print('\n\n')
    print(f'The size of the data frame : {df.size}\n')
    print(f'The shape of the data frame : {df.shape}\n')
    print('The transpose of Data Frame\n')
    print(df.T)
    print('\n\n')
    print('summary of the Data Frame\n')
    print(df.info(verbose = True))

def yearly_data(df):
    """This function takes data frame with years as columns and \
        converts country as columns and years as rows"""
    #slicing the df into only years columns
    y = df.loc[:,'1992':'2014']
    #changing values type from object to float
    y = y.astype(float)
    y['countries'] = df['Country Name']
    #transposing the data frame.
    y = y.T
    y.rename(columns = y.iloc[-1], inplace = True)
    y = y.drop(y.index[-1])
    #reset the index and making index as year column
    y = y.reset_index(drop = True)
    y = y.rename(columns={'index':'year'})
    return y


def preprocessing_data(df, column, drop_columns):
    # droping unused columns
    df =df.drop(columns=['Country Name','Country Code'])
    #transposing the data and assing the custom columns
    df_t =df.T
    df_t.columns = column
    if drop_columns:
        df_t = df_t.drop(columns=drop_columns)
    # reset the index and drop the index
    df_t = df_t.reset_index(drop = True)
    df_t = df_t.drop([0])
    df_t = df_t.astype(float)
    return df_t
    
def Clustermap_plot(df_t,image_name):
    """This function used to produce heat map of the given data frame
       seperating the data with respect to country. it takes color as
       parameter in order to change the colors of heat map."""   
       
    plt.figure()
    #plotting the cluster map using seaborn module
    sns.clustermap(df_t.corr(),annot=True, figsize=(16,10))
    plt.savefig(image_name,dpi=720)
    plt.title(image_name, loc='Left')
    plt.show()
    
def scattermap_plot(df_t,image_name):
    """This function used to produce heat map of the given data frame
       seperating the data with respect to country. it takes color as
       parameter in order to change the colors of heat map."""
       
    pd.plotting.scatter_matrix(df_t, figsize=(9.0, 9.0))
    plt.tight_layout() # helps to avoid overlap of labels
    plt.savefig(image_name,dpi=720)
    plt.show()
    
def norm(array):
    """ Returns array normalised to [0,1]. Array can be a numpy array
    or a column of a dataframe"""
    min_val = np.min(array)
    max_val = np.max(array)
    scaled = (array-min_val) / (max_val-min_val)
    return scaled

def norm_df(df):

    """
    Returns all columns of the dataframe normalised to [0,1] with the
    exception of the first (containing the names)
    Calls function norm to do the normalisation of one column, but
    doing all in one function is also fine.
    First, last: columns from first to last (including) are normalised.
    Defaulted to all. None is the empty entry. The default corresponds
    """
    # iterate over all numerical columns
    for col in df.columns[0:]: # excluding the first column
        df[col] = norm(df[col])
    return df

def kmeans_clusters(df):
    for ic in range(2, 9):
        # set up kmeans and fit
        kmeans = cluster.KMeans(n_clusters=ic)
        kmeans.fit(df)
        # extract labels and calculate silhoutte score
        labels = kmeans.labels_
        print (ic, skmet.silhouette_score(df, labels))
        
def cluster_plot(df,cluster_num):
    kmeans = cluster.KMeans(n_clusters= cluster_num)
    kmeans.fit(df)
    # extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    plt.figure(figsize=(6.0, 6.0))
    col = df.columns.tolist()
    # Individual colours can be assigned to symbols. The label l is used to the␣

    # l-th number from the colour table.
    plt.scatter(df[col[0]], df[col[1]], c=labels, cmap="Accent")
    for ic in range(cluster_num):
        xc, yc = cen[ic,:]
        plt.plot(xc, yc, "dk", markersize=10)
    plt.show()


pd_list = read_file('c0017547-d307-4149-af5e-579fb3c706de_Data.csv')

pd_countries = pd_list[0]
pd_years = pd_list[1]
#print(pd_countries['1995'])

#df_info(pd_countries)
col = ['GDP','GDP Per Capita','Exports','Imports','Agriculture','Industry'
        ,'Tax','Total Employment','Self-employed'
        ,'New bussiness','Reasearch']


pd_uk = pd_countries[pd_countries['Country Name']=='United Kingdom']
pd_uk = preprocessing_data(pd_uk, column=col, drop_columns=['New bussiness', 
                                          'Self-employed','GDP Per Capita'])
# Clustermap_plot(pd_uk, image_name='Cluster Map for UK')

# scattermap_plot(pd_uk, 'UK')

pd_uk_fit = pd_uk[['Imports','Exports']].copy()
pd_uk_fit = norm_df(pd_uk_fit)

print(pd_uk_fit.describe())



# kmeans_clusters(pd_uk_fit)
# cluster_plot(pd_uk_fit, cluster_num=8)
# cluster_plot(pd_uk_fit, cluster_num=3)


pd_uk_fit1 = pd_uk[['Total Employment','Tax']].copy()
pd_uk_fit1 = norm_df(pd_uk_fit1)

kmeans_clusters(pd_uk_fit1)
cluster_plot(pd_uk_fit1, cluster_num=2)
cluster_plot(pd_uk_fit1, cluster_num=5)


    
