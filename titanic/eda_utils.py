import pandas as pd
import seaborn as sns
import numpy as np
import tabulate as tbl
import matplotlib.pyplot as plt
from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler as SS
from sklearn.preprocessing import MinMaxScaler as MM
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer as LB
from sklearn.externals import joblib

import sklearn.decomposition
from sklearn.feature_extraction.text import CountVectorizer

def columns_unique(dataframe,columns):
    cache = {}
    for column in columns:
        cache[column] = dataframe[column].unique()
    return cache

def df_tabulate(df,line = None,title = None):
    if title is not None:
        print('----',title,'----')
    if line is None:
        print(tbl.tabulate(df, headers=df.columns, tablefmt='grid'))
    else:
        print(tbl.tabulate(df.head(line), headers=df.columns, tablefmt='grid'))

def dataframe_info(df):
    columns = df.columns.values
    dtypes = df.dtypes.values
    nannums = []
    nuni = []
    tmp = df.copy()
    for col in columns:
        tmp[col+'_nan'] = pd.isnull(df[col])
        nannums.append(len(tmp.loc[tmp[col+'_nan'] == True]))
        nuni.append(df[col].nunique())
    ndf = pd.DataFrame(np.array([columns,dtypes,nannums,nuni]).T)
    ndf.columns = ['key','type','nannum','nuni']
    return ndf

def corr_bar(df,feature,label):
    var = df.groupby([feature, label], as_index=False).size()
    var.unstack().plot(kind='bar', stacked=True, color=['lightpink', 'darkslateblue'])
    plt.xticks(rotation=360)
    plt.show()

def hist(df,feature,bins,label,lb_value):
    df[feature].plot.hist(bins=bins,color='darkslateblue')
    df.loc[df[label]==lb_value][feature].plot.hist(bins=bins,color='lightpink')
    plt.xlabel(feature)
    plt.ylabel('Size')
    plt.show()

def heatmap(df,columns):
    tmp = df[columns]
    plt.figure(figsize=(30,28))
    plt.title('Correlation of Features',y=1.05,size=20)
    sns.heatmap(tmp.astype(float).corr(),linewidths=0.1,vmax=1.0,
                square=True,linecolor='white',annot=True)
    plt.yticks(rotation=360)
    plt.savefig('eda/heatmap.png')
    plt.show()

def add_category_no(df,columns):
    columns_category_map = {}
    for col in columns:
        categories = df[col].unique()
        nos = [i for i in range(len(categories))]
        df[col+'_no'] = df[col].replace(categories, nos)
        print(categories)
        print(nos)
        columns_category_map[col] = categories
    return columns_category_map

def view_df_info(df):
    tif = dataframe_info(df)
    print('--------------------------------')
    df_tabulate(tif)
    discrete = tif.loc[(tif['type'] == 'object') & (tif['nuni'] < 10)]
    df_tabulate(discrete, title='discrete')
    nans = tif.loc[tif['nannum'] != 0]
    df_tabulate(nans, title='nans')
    print('--------------------------------')

def extract_ticket_preno(x):
    no = x
    if x == 'LINE':
         no = -1000
    if len(x.split(' '))>1:
        no = x.split(' ')[-1]
    return int(no)//1000

def cache_qts(df,col,quan):
    qts = []
    for i in range(1, quan):
        qts.append(df[col].quantile(i / quan))
    return qts

def fare_level(qts,x):
    for i in range(len(qts)):
        if x <= qts[i]:
            return i + 1
    return i + 1

def do_training():
    # 读取
    df = pd.read_csv('data/train.csv')
    df_tabulate(df,5)
    # 数据类型、空值、取值范围
    tif = dataframe_info(df)
    df_tabulate(tif)
    discrete = tif.loc[(tif['type']=='object') & (tif['nuni'] < 10)]
    df_tabulate(discrete,title='discrete')
    nans = tif.loc[tif['nannum']!= 0]
    df_tabulate(nans,title='nans')
    # 缺失值处理
    age_stat = df[['Pclass','Sex','Age']].groupby(['Pclass','Sex']).mean()
    print(age_stat)
    age_group = df.groupby(['Pclass','Sex'])['Age']
    df['Age'] = age_group.transform(lambda x:x.fillna(x.median()))
    df['Age_10'] = df['Age'].apply(lambda x: int(x//10))
    df['Age_child'] = df['Age'].apply(lambda x: 1 if x<=12 else 0)
    # df['Age'] = df['Age'].replace([np.nan],[0])
    df['Cabin'] = df['Cabin'].replace([np.nan],['NaN'])
    df['Cabin_nan'] =df['Cabin'].apply(lambda x:0 if x == 'NaN' else 1)
    var = df.groupby('Embarked',as_index=False).size()
    df['Embarked'] = df['Embarked'].replace([np.nan],[var.idxmax()])
    fare_group = df.groupby(['Pclass'])['Fare']
    df['Fare'] = fare_group.transform(lambda x: x.fillna(x.median()))
    # Emabarked/Sex与Survived关系直方分类图
    # label = 'Survived'
    # corr_features = ['Embarked','Sex','Pclass']
    # for cft in corr_features:
    #     corr_bar(df,cft,label)
    # 连续型特征与标签直方图
    # hist_features = ['Age','Fare','Parch','SibSp']
    # hist_bins_map = [16,20,7,7]
    # for i in range(len(hist_features)):
    #     hist(df,hist_features[i],hist_bins_map[i],label,1)
    # Parch与SibSp和
    df['Relation_sum'] = df['Parch']+df['SibSp']
    # SibSp和Age比率
    df['SibSpAge_rate'] = df.apply(lambda e: 0 if e['Age']==0 else e['SibSp']/e['Age']*100,axis=1)
    # Cabin增加特征
    df['Cabin_num'] = df['Cabin'].apply(lambda x: 0 if x=='NaN' else len(x.split(' ')))
    df['Cabin_prefix'] = df['Cabin'].apply(lambda x: x[0])
    # Name增加特征
    df['Name_len'] = df['Name'].apply(len)
    df['First_name'] = df['Name'].apply(lambda x:x.split(', ')[0])
    print(df['First_name'].nunique())
    df['Name_title'] = df['Name'].apply(lambda x:x.split(', ')[1].split('. ')[0])
    print(df['Name_title'].unique())
    title_stat = df[['Name_title','Survived']].groupby(['Name_title']).mean()
    print(title_stat)
    df['Name_title_level'] = df['Name_title']
    title_cates = [['the Countess','Lady','Don','Dona','Sir'],['Master','Jonkheer'],
                   ['Major' ,'Col','Capt','Rev' ,'Dr'],
                   ['Mrs','Mme','Ms' ],['Miss','Mlle'],['Mr']]
    for i in range(len(title_cates)):
        df['Name_title_level'] = df['Name_title_level'].replace(title_cates[i],6-i)
    print(df['Name_title_level'].unique())
    # & (df['Age_child']==0)
    has_family = df[(df['Relation_sum']!=0)].groupby(['Sex','First_name'])['Survived']
    df['Family_effect'] = has_family.transform(lambda x:x.median())
    df['Family_effect'] = df['Family_effect'].replace([np.nan,0],[0,-1])
    print(df['Family_effect'].nunique())
    df['Ticket_has_prefix'] = df['Ticket'].apply(lambda x:1 if len(x.split(' '))>1 else 0)
    df['Ticket_prefix'] = df['Ticket'].apply(lambda x:x.split(' ')[0] if len(x.split(' '))>1 else 'NaN')
    df['Ticket_preno'] = df['Ticket'].apply(extract_ticket_preno)
    df['Ticket_first'] = df['Ticket'].apply(lambda x:x[0])
    map = dict(df['Ticket'].value_counts())
    print(map)
    df['Ticket_common'] = df['Ticket'].apply(lambda x:map[x])
    df['Fare_avg'] = df['Fare']/df['Ticket_common']
    qts = cache_qts(df,'Fare_avg',10)
    # df[df['Fare_avg']<=20]['Fare_avg'].hist()
    # plt.show()
    df['Fare_level'] = df['Fare_avg'].apply(lambda x: fare_level(qts,x))
    print(df['Fare_level'].value_counts())
    # 离散序号编码
    le = LabelEncoder()
    lb_features = ['Sex','Embarked','Cabin_prefix','Name_title','Ticket_prefix','Ticket_first']
    for feature in lb_features:
        le = le.fit(df[feature])
        df[feature+'_no'] = le.transform(df[feature])
    # 相关性热力图
    heat_features = ['Survived','Pclass','Age','SibSp','Parch','Fare',
                     'Sex_no','Embarked_no','Cabin_num','Cabin_prefix_no',
                     'Name_title_no','Ticket_has_prefix','Ticket_prefix_no',
                     'Ticket_preno','Ticket_common','Ticket_first_no','Fare_level',
                     'Relation_sum','SibSpAge_rate','Family_effect','Fare_avg',
                     'Name_len','Age_10','Age_child','Cabin_nan','Name_title_level']
    # heatmap(df,heat_features)
    # print(len(heat_features))
    # df_tabulate(df[heat_features],5)
    # df列
    df_tabulate(df,5)
    tif = dataframe_info(df)
    df_tabulate(tif)
    # tif.to_csv('eda/eda_info.csv')
    # df.to_csv('eda/eda_result.csv')
    # 生成特征
    ftdf = pd.read_csv('eda/eda_feature.csv')
    print(ftdf.columns)
    df_tabulate(ftdf)
    label_df = df['Survived']
    # drop columns
    # df.drop('Survived')
    ftdf['todo'] = ftdf['todo'].replace(['drop','ddrop','label'],'drop')
    to_drop_ft = ftdf[ftdf['todo'] == 'drop']
    drop_columns = list(to_drop_ft['key'].values)
    print('drop:',drop_columns)
    train_df = df.drop(drop_columns,axis=1)
    df_tabulate(train_df,5)
    # dummies
    dummy_ft = ftdf[ftdf['todo'] == 'dummy']
    dummy_columns = list(dummy_ft['key'].values)
    print('dummy',dummy_columns)
    train_df = pd.get_dummies(train_df, columns = dummy_columns, dummy_na=False)
    df_tabulate(train_df,5)
    # df = pd.concat([df,dm],axis=1)
    # dummies_features = []
    # for col in dummies_columns:
    #     dmfts = [col+'_'+ct for ct in df[col].unique()]
    #     dummies_features += dmfts

    # 变换 np array
    y_set = np.array(label_df).reshape(-1,1).T
    print(y_set.shape)
    x_set = np.array(train_df)
    print(x_set.shape)
    # # Train the logistic regression classifier
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(x_set, y_set.ravel())
    # Print accuracy
    LR_predictions = clf.predict(x_set).reshape(-1,1)
    print(round(clf.score(x_set,y_set.ravel())*100,4))
    print ('Accuracy of logistic regression: %d ' % float((np.dot(y_set,LR_predictions) + np.dot(1-y_set,1-LR_predictions))/float(y_set.size)*100) +
           '% ' + "(percentage of correctly labelled datapoints)")

def pivot_eda():
    df = pd.read_csv('data/train.csv')
    print(df.info())
    tb = pd.pivot_table(df,index=['Sex'],values=['Survived'])
    print(tb)

pivot_eda()
# do_training()
