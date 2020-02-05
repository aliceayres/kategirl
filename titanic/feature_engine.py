import pandas as pd
import numpy as np
import sklearn.decomposition
import matplotlib.pyplot as plt
import eda_utils as eda
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score,f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import xgboost as xgb
import warnings
import sklearn.decomposition
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import zipfile

def zip_result(filename):
    z = zipfile.ZipFile('eda/result.zip', 'w')
    z.write(filename,'result.csv')
    z.close()

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

def fare_level_bak(qts,x):
    for i in range(len(qts)):
        if x <= qts[i]:
            return i + 1
    return i + 1

def fare_level(x):
    fare = round(x,2)
    if fare == 0:
        return 0
    elif fare <=7.23:
        return 1
    elif fare <=7.75:
        return 2
    elif fare <=7.9:
        return 3
    elif fare <=8.05:
        return 4
    elif fare <=10.5:
        return 5
    elif fare <=13:
        return 6
    elif fare <= 26.55:
        return 7
    return 8

def highlight(value):
    if value >= 0.5:
        style = 'background-color: palegreen'
    else:
        style = 'background-color: pink'
    return style

def feature_generate(trainfile,testfile):
    train_df = pd.read_csv(trainfile)
    test_df = pd.read_csv(testfile)
    test_df['Survived'] = 2
    df = pd.concat([train_df,test_df],axis=0,sort=True)
    # print('concat train & test:')
    # print(df)
    print(pd.pivot_table(df, values='Survived', index=['Sex']))
    print(df.info())
    # 观察数据情况
    # eda.view_df_info(df)
    # 缺失值处理
    # add : name unique title
    df['Name_title'] = df['Name'].apply(lambda x: x.split(', ')[1].split('. ')[0])
    # print(df['Name_title'].unique())
    # add : name title level
    df['Is_Master'] = df['Name_title'].apply(lambda x:1 if x=='Master' else 0)
    df['Name_title_level'] = df['Name_title']
    title_cates = [['the Countess', 'Lady', 'Don', 'Dona', 'Sir'], ['Master', 'Jonkheer'],
                   ['Major', 'Col', 'Capt', 'Rev', 'Dr'],
                   ['Mrs', 'Mme', 'Ms'], ['Miss', 'Mlle'], ['Mr']]
    for i in range(len(title_cates)):
        df['Name_title_level'] = df['Name_title_level'].replace(title_cates[i], 6 - i)
    # age = pclass & sex median
    age_group = df.groupby(['Name_title_level', 'Sex'])['Age']
    df['Age'] = age_group.transform(lambda x: x.fillna(x.median()))
    # cabin = 'NaN'
    df['Cabin'] = df['Cabin'].replace([np.nan], ['NaN'])
    # embarked = mode
    var = df.groupby('Embarked', as_index=False).size()
    df['Embarked'] = df['Embarked'].replace([np.nan], [var.idxmax()])
    # fare = pclass median
    fare_group = df.groupby(['Pclass'])['Fare']
    df['Fare'] = fare_group.transform(lambda x: x.fillna(x.median()))
    # 增加其他特征
    # add : passengerId/10 /100
    df['PassengerId_10'] = df['PassengerId'].apply(lambda x: int(x // 10))
    df['PassengerId_100'] = df['PassengerId'].apply(lambda x: int(x // 100))
    # add : age/10
    df['Age_5'] = df['Age'].apply(lambda x: int(x // 5))
    df['Age_10'] = df['Age'].apply(lambda x: int(x // 10))
    df['Age_15'] = df['Age'].apply(lambda x: int(x // 15))
    df['Age_20'] = df['Age'].apply(lambda x: int(x // 20))
    # add : age is child
    df['Age_child'] = df['Age'].apply(lambda x: 1 if x <= 12 else 0)
    # add : pclass and sex
    df['Pclass_sex'] = df.Sex + "_" + df.Pclass.map(str)
    df['Pclass_embarked'] = df.Embarked + "_" + df.Pclass.map(str)
    # df['Title_sex'] = df.Sex + "_" + df.Name_title_level.map(str)
    # add : cabin is nan
    df['Cabin_nan'] = df['Cabin'].apply(lambda x: 0 if x == 'NaN' else 1)
    # add : Parch + SibSp
    df['Relation_sum'] = df['Parch']+df['SibSp']
    df['Singleton'] = df['Relation_sum'].map(lambda s: 1 if s == 0 else 0)
    df['SmallFamily'] = df['Relation_sum'].map(lambda s: 1 if 1 <= s <= 4 else 0)
    df['LargeFamily'] = df['Relation_sum'].map(lambda s: 1 if 5 <= s else 0)
    # add : SibSp和Age比率
    df['SibSpAge_rate'] = df.apply(lambda e: 0 if e['Age']==0 else e['SibSp']/e['Age']*100,axis=1)
    # add : Parch和Age比率
    df['ParchAge_rate'] = df.apply(lambda e: 0 if e['Age'] == 0 else e['Parch'] / e['Age'] * 100, axis=1)
    # add : cabin num
    df['Cabin_num'] = df['Cabin'].apply(lambda x: 0 if x =='NaN' else len(x.split(' ')))
    # add : cabin first letter
    df['Cabin_prefix'] = df['Cabin'].apply(lambda x: x[0])
    # add : name length
    df['Name_len'] = df['Name'].apply(len)
    # add : family survived effect & test has no label
    # & (df['Age_child']==0)
    # df['First_name'] = df['Name'].apply(lambda x: x.split(', ')[0])
    # le = LabelEncoder()
    # le = le.fit(df['First_name'])
    # df['First_name_no'] = le.transform(df['First_name'])
    # has_family = df[(df['Relation_sum']!=0)].groupby(['Sex','First_name'])['Survived']
    # df['Family_effect'] = has_family.transform(lambda x:x.median())
    # df['Family_effect'] = df['Family_effect'].replace([np.nan,0],[0,-1])
    # df = df.drop(['First_name'],axis = 1)
    # add : ticket has prefix letters
    df['Ticket_has_prefix'] = df['Ticket'].apply(lambda x:1 if len(x.split(' '))>1 else 0)
    # add : ticket prefix letters
    df['Ticket_prefix'] = df['Ticket'].apply(lambda x:x.split(' ')[0] if len(x.split(' '))>1 else 'NaN')
    # add : ticket prefix 3-4 digit
    df['Ticket_preno'] = df['Ticket'].apply(extract_ticket_preno)
    # add : ticket prefix 1 letter or digit @@@@
    df['Ticket_first'] = df['Ticket'].apply(lambda x:x[0])
    df['Imp_ticket_fst_h'] = df['Ticket_first'].apply(lambda x: 1 if str(x) in ['1','3','2'] else 0)
    df['Imp_ticket_fst_m'] = df['Ticket_first'].apply(lambda x: 1 if str(x) in ['S', '7', '2'] else 0)
    df['Imp_ticket_fst_l'] = df['Ticket_first'].apply(lambda x: 1 if str(x) in ['P', '4', 'C','A', 'W', '6'] else 0)
    # add : ticket is group
    map = dict(df['Ticket'].value_counts())
    df['Ticket_common'] = df['Ticket'].apply(lambda x:map[x])
    # add : average fare
    df['Fare_avg'] = df['Fare']/df['Ticket_common']
    df['Fare_avg_7'] = df['Fare_avg'] // 7
    df['Fare_avg_13'] = df['Fare_avg'] // 13
    # df['Fare_avg_26'] = df['Fare_avg'] // 26.55
    # df['Fare_avg_int'] =  df['Fare_avg'].apply(lambda x: round(x,2))
    # df_fare_group = df.groupby(['Fare_avg_int']).size()
    # print(df_fare_group)
    # df_fare_group.plot(kind='barh', figsize=(25, 25))
    # plt.show()
    qts = cache_qts(df,'Fare_avg',10)
    df['Fare_level'] = df['Fare_avg'].apply(lambda x:fare_level(x))
    # add : important ticket pre no
    df['Imp_ticket_pre'] = df['Ticket_preno'].apply(lambda x:1 if x in [349,347,113,17,2,382,244,345,367,19,1] else 0)
    # One-hot编码
    # dummies 'Title_sex','Name_title_level','Ticket_preno','Fare_level','Pclass'
    dummy = True
    dummy_columns = ['Sex', 'Embarked', 'Cabin_prefix','Pclass_embarked',
                     'Ticket_has_prefix', 'Ticket_prefix', 'Ticket_first','Pclass_sex']  #
    if dummy is True:
        df = pd.get_dummies(df, columns=dummy_columns, dummy_na=False)
    else:
        # 离散序号编码
        le = LabelEncoder()
        for feature in dummy_columns:
            le = le.fit(df[feature])
            df[feature + '_no'] = le.transform(df[feature])
    # view_df_info(df)
    # 删除列 留下ID列和Label列
    drop_columns = ['Name', 'Ticket', 'Cabin', 'Ticket_preno', 'Name_title']
    df = df.drop(drop_columns, axis=1)
    # eda.view_df_info(df)
    df.to_csv('feature/original.csv')
    return df

def load_data(reload):
    if reload is True:
        df = feature_generate('data/train.csv','data/test.csv')
    else:
        df = pd.read_csv('feature/original.csv')
    id_col = 'PassengerId'
    label_col = 'Survived'
    train = df.iloc[:891, :]
    label = train[label_col]
    test = df.iloc[891:, :]
    id_df = test[id_col]
    train = train.drop([label_col], axis=1)
    test = test.drop([label_col], axis=1)
    return train,label,test,id_df

def transform_matrix(train,label,test):
    y_set = np.array(label).reshape(-1, 1).T
    x_set = np.array(train)
    x_test = np.array(test)
    assert (y_set.shape[1] == x_set.shape[0])
    assert (x_set.shape[1] == x_test.shape[1])
    # 标准化
    scaler = StandardScaler()
    x_set = scaler.fit(x_set).transform(x_set)
    x_test = scaler.fit(x_test).transform(x_test)
    return x_set, y_set, x_test

def model_generate(x,y):
    lrcv = sklearn.linear_model.LogisticRegressionCV()
    lr = sklearn.linear_model.LogisticRegression(tol=1e-6)
    xgbt_default = {'eta': 0.01,
                    'n_estimators': 1000,
                    'gamma': 0,
                    'max_depth': 8,
                    'min_child_weight': 1,
                    'colsample_bytree': 1,
                    'colsample_bylevel': 1,
                    'subsample': 1,
                    'reg_lambda': 1,
                    'reg_alpha': 0,
                    'seed': 1024,
                    'objective': 'binary:logistic',
                    'eval_metric': 'error'}
    xgbt = XGBClassifier(**xgbt_default)
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=15)
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier(max_depth=12,max_features='sqrt',min_samples_leaf=3,
                                min_samples_split=2,n_estimators=100,random_state=6)
    params = {'max_depth' : [4, 6, 8,10,12],
                 'n_estimators': [100,90,80,70,60,50, 40,30,20,10]}
    model = rf
    search_open = False
    if search_open is True:
        gs = GridSearchCV(model, params, n_jobs=-1, cv=5, verbose=1,scoring='accuracy')
        gs.fit(x, y.ravel())
        print(gs.best_score_)  # 最好的得分
        print(gs.best_params_)  # 最好的参数
        print(sklearn.metrics.SCORERS.keys())
    else:
        model.fit(x, y.ravel())
    return model

def predict_test(x,y,x_test,model):
    model.fit(x,y.ravel())
    # Train与Test比较评分(欠拟合与过拟合)
    cv = sklearn.model_selection.cross_validate(model, x, y.ravel(), cv=5, scoring='accuracy')
    for item in cv.items():
        print(item[0] + ':', round(item[1].mean(), 5))
    return model.predict(x_test).reshape(-1,1)

def submission_generate(id_df,y_pred,csv):
    data_list = map(lambda x: x[0], y_pred)
    result_series = pd.Series(data_list)
    id_df = id_df.reset_index(drop=True)
    rt = pd.concat([id_df, result_series], axis=1)
    # eda.df_tabulate(rt)
    rt.columns = ['PassengerId','Survived']
    rt.to_csv(csv,index=False)
    zip_result(csv)

def run_titanic():
    reload = True
    train, label, test, id_df = load_data(reload)
    x_set,y_set,x_test = transform_matrix(train,label,test)
    model = model_generate(x_set,y_set)
    # 随机森林的特征重要程度
    features = pd.DataFrame()
    features['feature'] = train.columns
    features['importance'] = model.feature_importances_
    print(x_set.shape)
    print(model.feature_importances_.shape)
    features['importance'] = features['importance']*1000
    features = features[features['importance']>=3.5]
    features.sort_values(by=['importance'], ascending=False, inplace=True)
    eda.df_tabulate(features)
    # 显示图
    # features.set_index('feature', inplace=True)
    # features.plot(kind='barh', figsize=(25, 25))
    # plt.show()
    y_test = predict_test(x_set,y_set,x_test,model)
    print(y_test.shape)
    csv = 'eda/result.csv'
    submission_generate(id_df,y_test,csv)

def doing_model():
    reload = True
    train, label, test, id_df = load_data(reload)
    x_set, y_set, x_test = transform_matrix(train, label, test)
    y_set = y_set.T
    # print(x_set.shape)
    # print(y_set.shape)
    # 模型生成及网格搜索 GridSearchCV
    model = model_generate(x_set,y_set)
    k = 5
    # Train与Test比较评分(欠拟合与过拟合)
    cv = sklearn.model_selection.cross_validate(model, x_set, y_set.ravel(), cv=k, scoring='accuracy')
    for item in cv.items():
        print(item[0]+':',round(item[1].mean(),5))
    # 综合评分 k-fold score
    # scores = sklearn.model_selection.cross_val_score(model, x_set, y_set.ravel(), cv=k, scoring='accuracy')
    # print('LogisticRegressionCV:',round(scores.mean(),5))

    # train/test split
    # train_X, dev_X, train_y, dev_y = sklearn.model_selection.train_test_split(x_set,y_set,test_size=0.3,random_state=6)
    # clf = sklearn.linear_model.LogisticRegressionCV()
    # clf.fit(train_X, train_y.ravel())
    # print(round(clf.score(train_X, train_y.ravel()), 5))
    # print(round(clf.score(dev_X, dev_y.ravel()), 5))

    # k-fold
    # kf = sklearn.model_selection.KFold(n_splits=k, shuffle=False)
    # clf = sklearn.linear_model.LogisticRegressionCV()
    # dev_scores = 0
    # for train_index, test_index in kf.split(x_set):
    #     begin = test_index[0]
    #     end = test_index[-1]
    #     k_dev_x = x_set[begin:end+1,:]
    #     k_dev_y = y_set[begin:end+1,:]
    #     k_train_x = np.delete(x_set, test_index, axis=0)
    #     k_train_y = np.delete(y_set, test_index, axis=0)
    #     # print(k_dev_x.shape)
    #     # print(k_dev_y.shape)
    #     # print(k_train_x.shape)
    #     # print(k_train_y.shape)
    #     clf.fit(k_train_x, k_train_y.ravel())
    #     dev_scores += clf.score(k_dev_x, k_dev_y.ravel())
    #     # print(round(clf.score(k_train_x, k_train_y.ravel()), 5))
    #     # print(round(clf.score(k_dev_x, k_dev_y.ravel()), 5))
    # print(dev_scores/k)


if __name__=='__main__':
    warnings.filterwarnings(action='ignore',category=DeprecationWarning)
    run_titanic()
    # doing_model()