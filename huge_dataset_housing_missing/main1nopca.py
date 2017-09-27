import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from xgboost import XGBRegressor as xgb
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import time



def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings

            # Print current column type
            #print("******************************")
            #print("Column: ", col)
            #print("dtype before: ", props[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            if(props[col].isnull().all()):
                me=0
            else:
                me = props[col].mean()
            mn = props[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(me, inplace=True)

                # test if column can be converted to an integer
            asint = props[col].fillna(me).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

                        # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

            # Print new column type
            #print("dtype after: ", props[col].dtype)
            #print("******************************")

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return props, NAlist

def str2int( ds ,target):
    storenull = ds[target].isnull()
    enc = LabelEncoder( )
    ds[target]= enc.fit_transform( ds[target] )
    ds.loc[storenull,target] = np.nan
    #print("ds:",ds[target])
    return ds

def fillna_knn(ds,target) :
    miss = ds[target].isnull()
    notmiss = ~miss

    X = ds.loc[notmiss,["latitude","longitude"]]
    Y = ds.loc[notmiss,target]

    clf = KNeighborsClassifier(n_neighbors=5, weights='uniform',n_jobs=-1)
    clf.fit(X, Y)

    pred = clf.predict(ds.loc[miss,["latitude","longitude"]])
    ds.loc[miss,target] = pred
    return ds

url = "/home/netik/Desktop/zillow"
#print(time.ctime(time1))

train = pd.read_csv(url+"/train_2016_v2.csv", parse_dates=["transactiondate"])
prop = pd.read_csv(url+"/properties_2016.csv")
i =0
for col in prop.columns:
    if prop[col].dtype == "float64" :
        prop[col] = prop[col].astype(np.float32)

#train["logerror"],l = reduce_mem_usage(train["logerror"])
#print("train",train.shape,train.dtypes)
#print("properties:",prop.shape)

#print(prop.isnull().sum())
thresh = len(prop)*0.01
#print("columns available:",prop.columns)
#prop = prop.dropna(thresh = thresh,axis = 1)

#print("properties after merge after dropna 98%:",prop.shape)
#print(prop.dtypes)
print(prop.isnull().sum()/len(prop) *100,prop.isnull().sum(),prop.dtypes)

#trying handling the null for geographical datas
propmiss = prop["latitude"].isnull()
#print(prop)
#print("propmiss:",propmiss)
propnull_ll = prop[prop["latitude"].isnull()]
#print("propnull_ll:",propnull_ll)
prop.dropna(axis=0, inplace=True, subset=["latitude", "longitude"])

prop = str2int(prop,"propertyzoningdesc")
prop["propertyzoningdesc"] = fillna_knn(
    prop,"propertyzoningdesc")

prop = str2int(prop,"propertycountylandusecode")
prop["propertycountylandusecode"] = fillna_knn(
    prop,"propertycountylandusecode")

prop = str2int(prop,"regionidcity")
prop["regionidcity"] = fillna_knn(
    prop,"regionidcity")

prop = str2int(prop,"regionidneighborhood")
prop["regionidneighborhood"] = fillna_knn(
    prop,"regionidneighborhood")

prop = str2int(prop,"regionidzip")
prop["regionidzip"] = fillna_knn(
    prop,"regionidzip")

prop = str2int(prop,"yearbuilt")
prop["yearbuilt"] = fillna_knn(
    prop,"yearbuilt")


prop = str2int(prop,"lotsizesquarefeet")
prop["lotsizesquarefeet"] = fillna_knn(
    prop,"lotsizesquarefeet")

prop = str2int(prop,"buildingqualitytypeid")
prop["buildingqualitytypeid"] = fillna_knn(
    prop,"buildingqualitytypeid")


prop = prop.append(propnull_ll).sort_index()

#Average structuretaxvaluedollarcnt by city
group = prop.groupby('regionidcity')['structuretaxvaluedollarcnt'].aggregate('mean').to_dict()
prop['N-Avg-structuretaxvaluedollarcnt'] = prop['regionidcity'].map(group)

#Deviation away from average
prop['N-Dev-structuretaxvaluedollarcnt'] = abs((prop['structuretaxvaluedollarcnt'] - prop['N-Avg-structuretaxvaluedollarcnt']))/prop['N-Avg-structuretaxvaluedollarcnt']



#Number of properties in the zip
zip_count = prop['regionidzip'].value_counts().to_dict()
prop['N-zip_count'] = prop['regionidzip'].map(zip_count)

#Number of properties in the city
city_count = prop['regionidcity'].value_counts().to_dict()
prop['N-city_count'] = prop['regionidcity'].map(city_count)

#Ratio of tax of property over parcel
prop['N-ValueRatio'] = prop['taxvaluedollarcnt']/prop['taxamount']

#TotalTaxScore
prop['N-TaxScore'] = prop['taxvaluedollarcnt']*prop['taxamount']

prop['N-ValueProp'] = prop['structuretaxvaluedollarcnt']/prop['landtaxvaluedollarcnt']
#proportion of living area
prop['N-LivingAreaProp'] = prop['calculatedfinishedsquarefeet']/prop['lotsizesquarefeet']
print("labelencoding")
lbl = LabelEncoder()
for col in prop.columns:
    if(prop[col].isnull().any()):
        s = col+"null"
        prop[s] = prop[col].isnull()
        prop[s] = prop[s].astype(np.int8)
    if prop[col].dtype == "object" :
        prop[col] = lbl.fit_transform(list(prop[col].values))
print(prop.shape)

print("label encoding done")
#print(prop[prop.columns[32]],prop[prop.columns[34]])
#lbl.fit(list(prop["propertyzoningdesc"].values))
#prop["propertyzoningdesc"] = lbl.transform(list(prop["propertyzoningdesc"].values))
#enc = OneHotEncoder(categorical_features=[32,34],sparse=False)


prop,NAlist = reduce_mem_usage(prop)
print("reducing mem usage")
pid = prop["parcelid"]
print(pid.shape)
prop.drop(["parcelid"],axis =1,inplace=True)
print("scaling")
print("minmaxscalr strt")
prop = MinMaxScaler().fit_transform(prop)
print("minmaxscaler doone")
print ("prop shape before pca:", prop.shape)
#print("pca strt")
#pca = PCA(n_components = 95,copy = False)
#prop = pca.fit_transform(prop)

#prop = pd.DataFrame(prop)
#print("pca done")
prop = pd.concat([prop,pid],axis = 1)
#Cumulative Variance explains
#var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
#plt.plot(var1)
#plt.show()
#print ("prop shape after pca:", prop.shape)
prop,NAlist = reduce_mem_usage(prop)
del  NAlist,lbl , pid
gc.collect()
#print(prop.dtypes,"nalist:",NAlist)
testds = prop
prop = pd.merge(train,prop,on = "parcelid",how = "left")
print(prop.shape)
#print("properties after merge:",prop.shape)
del train
gc.collect()
#print(prop.isnull().sum()/len(prop))


prop["transactionyear"] = prop["transactiondate"].dt.year
prop["transactionmonth"] = prop["transactiondate"].dt.month
prop = prop.drop(["transactiondate"],axis=1)

#print(prop.parcelid.value_counts().max())



#prop.fillna(prop.mean(),inplace = True)
prop=prop[ prop.logerror > -0.4 ]
prop=prop[ prop.logerror < 0.418 ]
x_train = prop
x_train = x_train.drop(["logerror","parcelid"],axis=1)
cols = x_train.columns.values.tolist()

#enc = OneHotEncoder(categorical_features=[0,1,2,3,4,10,11,12,13,15,16,20,21,22,23,24,
 #                                         26,27,28,29,30,31,32,35],sparse=False)


y_train = prop["logerror"].values
del prop


#print(prop.head(20))
#floatcols = {"logerror","fips","garagetotalsqft","lotsizesquarefeet",
 #            "calculatedfinishedsquarefeet","finishedsquarefeet12","rawcensustractandblock",
  #           "unitcnt","structuretaxvaluedollarcnt",
   #          "taxvaluedollarcnt","landtaxvaluedollarcnt","taxamount","censustractandblock"}

#i=0
#for cols in floatcols:
 #   i = i+1
  #  print(i,".",cols, ":", prop[cols].max(axis = 0),",",prop[cols].min(axis = 0))

#for col in floatcols:
 #   prop.plot(x="parcelid",y=col,kind="scatter")
#plt.show()

#reg = svm.SVR()

#T_train_xgb = xgb.DMatrix(x_train, y_train)
gbm = xgb(objective="reg:linear")

print("fitting strt")
gbm = gbm.fit(x_train,y_train)

#print(x_train.shape,x_train.dtypes)

#print(cols)
#reg.fit(x_train,y_train)

print("fitting done")
del x_train,y_train
gc.collect()

submission = pd.read_csv(url+"/sample_submission.csv")
date = pd.to_datetime(str(submission.columns[2]),format = "%Y%m")
#testds = pd.read_csv(url+"/properties_2016.csv")
#for col in testds.columns:
#    if testds[col].dtype == "float64" :
#        testds[col] = testds[col].astype(np.float32)
#lbl.fit(list(testds["propertycountylandusecode"].values))
#testds["propertycountylandusecode"] = lbl.transform(list(testds["propertycountylandusecode"].values))
#lbl.fit(list(testds["propertyzoningdesc"].values))
#testds["propertyzoningdesc"] = lbl.transform(list(testds["propertyzoningdesc"].values))
#testds = enc.transform(testds)

#testds = testds.dropna(thresh = thresh,axis = 1)

#testds["hashottuborspa"].fillna(False, inplace=True)
#testds["hashottuborspa"] = testds["hashottuborspa"].astype(int)

testds["transactionyear"] = date.year
testds["transactionmonth"] = date.month
testds = testds[cols]
print("testds shape:",testds.shape )
#testds = reduce_mem_usage(testds)#testds.fillna(testds.mean())

#print(testds.shape,testds.dtypes)
gc.collect()

print("lprdeict strt")
for c in submission.columns[submission.columns != "ParcelId"] :
    date = pd.to_datetime(str(c),format = "%Y%m")
    testds["transactionyear"] = date.year
    testds["transactionmonth"] = date.month
    print("predivting for:",str(c),str(date) )
    prediction = gbm.predict(testds)
    prediction = np.round(prediction,decimals=4)
    print(prediction)
    submission[c] = prediction
    print(submission[c])
    print(str(c),"done")
del testds
gc.collect()
print("predict done:")
print(submission.shape)
print(submission.head(20))
print("writing to csv")
submission.to_csv(url+"/noPCA.csv",index = False)
print("wrintg to csv done")




