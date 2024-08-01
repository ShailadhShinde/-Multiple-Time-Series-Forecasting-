# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-08-01T14:40:27.991984Z","iopub.execute_input":"2024-08-01T14:40:27.992457Z","iopub.status.idle":"2024-08-01T14:40:28.991348Z","shell.execute_reply.started":"2024-08-01T14:40:27.992419Z","shell.execute_reply":"2024-08-01T14:40:28.990288Z"}}
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-08-01T14:40:31.705512Z","iopub.execute_input":"2024-08-01T14:40:31.706090Z","iopub.status.idle":"2024-08-01T14:40:38.576918Z","shell.execute_reply.started":"2024-08-01T14:40:31.706054Z","shell.execute_reply":"2024-08-01T14:40:38.575804Z"}}

train = pd.read_csv(r"/kaggle/input/store-sales-time-series-forecasting/train.csv",
                   usecols=[1,2,3,4,5],parse_dates=['date'],
                   converters={'sales': lambda u: np.log1p(float(u)) if float(u) > 0 else 0},
                  )
test = pd.read_csv(r"/kaggle/input/store-sales-time-series-forecasting/test.csv",
                  usecols=[1,2,3,4],parse_dates=['date']
                  )

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-08-01T14:40:38.581218Z","iopub.execute_input":"2024-08-01T14:40:38.581571Z","iopub.status.idle":"2024-08-01T14:40:42.298115Z","shell.execute_reply.started":"2024-08-01T14:40:38.581541Z","shell.execute_reply":"2024-08-01T14:40:42.296982Z"}}
promo_train=train.set_index(['store_nbr','family','date'])['onpromotion'].unstack().fillna(0)
promo_test=test.set_index(['store_nbr','family','date'])['onpromotion'].unstack().fillna(0)
promo=pd.concat([promo_train,promo_test],axis=1)

sales_train=train.set_index(['store_nbr','family','date'])['sales'].unstack().fillna(0)

family=sales_train.groupby('family')[sales_train.columns].sum()
family_promo=promo.groupby('family')[promo.columns].sum()

store_family=sales_train.reset_index()
store_family_index=store_family[['store_nbr','family']]
store_family=store_family.groupby(['store_nbr','family'])[sales_train.columns].sum()

store_family_promo=promo.reset_index()

stores = pd.read_csv(r"/kaggle/input/store-sales-time-series-forecasting/stores.csv").set_index('store_nbr')
le = LabelEncoder()

stores['city'] = le.fit_transform(stores['city'].values)
stores['state'] = le.fit_transform(stores['state'].values)
stores['type'] = le.fit_transform(stores['type'].values)

stores=stores.reindex(sales_train.index.get_level_values(0))

#promo=pd.concat([promo_train,promo_test],axis=1)
#del promo_train,promo_test

first_date = '2013-01-01'
val_start=pd.to_datetime('2017-7-25')

test_start=pd.to_datetime('2017-8-16')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-08-01T14:40:46.977416Z","iopub.execute_input":"2024-08-01T14:40:46.977948Z","iopub.status.idle":"2024-08-01T14:40:46.984083Z","shell.execute_reply.started":"2024-08-01T14:40:46.977904Z","shell.execute_reply":"2024-08-01T14:40:46.982840Z"}}
def get_timespan(df,date,minusdays,periods,freq='D'):
    return df[pd.date_range(date-timedelta(days=minusdays) ,periods=periods, freq=freq)]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-08-01T14:40:42.311135Z","iopub.execute_input":"2024-08-01T14:40:42.311598Z","iopub.status.idle":"2024-08-01T14:40:42.333410Z","shell.execute_reply.started":"2024-08-01T14:40:42.311559Z","shell.execute_reply":"2024-08-01T14:40:42.332102Z"}}
def prepare_dataset(df,promo,date,columnname=None,is_train=True):
    
   
    X={
        "promo_3_bef": get_timespan(promo, date, 3, 3).sum(axis=1).values,         
        "promo_7_bef": get_timespan(promo, date, 7, 7).sum(axis=1).values,         
        "promo_14_bef": get_timespan(promo, date, 14, 14).sum(axis=1).values,         
        "promo_60_bef": get_timespan(promo, date, 60, 60).sum(axis=1).values,
        "promo_140_bef": get_timespan(promo, date, 140, 140).sum(axis=1).values,
        "promo_3_aft": get_timespan(promo, date + timedelta(days=16), 15, 3).sum(axis=1).values,
        "promo_7_aft": get_timespan(promo, date + timedelta(days=16), 15, 7).sum(axis=1).values,
        "promo_14_aft": get_timespan(promo, date + timedelta(days=16), 15, 14).sum(axis=1).values,
        "promo_mean_3_bef": get_timespan(promo, date, 3, 3).mean(axis=1).values,         
        "promo_mean_7_bef": get_timespan(promo, date, 7, 7).mean(axis=1).values,         
        "promo_mean_14_bef": get_timespan(promo, date, 14, 14).mean(axis=1).values,         
        "promo_mean_60_bef": get_timespan(promo, date, 60, 60).mean(axis=1).values,
        "promo_mean_140_bef": get_timespan(promo, date, 140, 140).mean(axis=1).values,
        "promo_mean_3_aft": get_timespan(promo, date + timedelta(days=16), 15, 3).mean(axis=1).values,
        "promo_mean_7_aft": get_timespan(promo, date + timedelta(days=16), 15, 7).mean(axis=1).values,
        "promo_mean_14_aft": get_timespan(promo, date + timedelta(days=16), 15, 14).mean(axis=1).values,
        
    }
    for i in [3,7,14,30,60,140]:
        tmpdf=get_timespan(df,date + timedelta(days=-7),i,i)
        X['diff_%s_mean_2' %i]=tmpdf.diff(axis=1).mean(axis=1).values
        X['mean_%s_decay_2' % i] = (tmpdf * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X['mean_%s_2' % i] = tmpdf.mean(axis=1).values
        X['median_%s_2' % i] = tmpdf.median(axis=1).values
        X['min_%s_2' % i] = tmpdf.min(axis=1).values
        X['max_%s_2' % i] = tmpdf.max(axis=1).values
        X['std_%s_2' % i] = tmpdf.std(axis=1).values
    
    for i in [3,7,14,30,60,140]:
        tmpdf=get_timespan(df,date,i,i)
        X['diff_%s_mean' %i]=tmpdf.diff(axis=1).mean(axis=1).values
        X['mean_%s_decay' % i] = (tmpdf * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X['mean_%s' % i] = tmpdf.mean(axis=1).values
        X['median_%s' % i] = tmpdf.median(axis=1).values
        X['min_%s' % i] = tmpdf.min(axis=1).values
        X['max_%s' % i] = tmpdf.max(axis=1).values
        X['std_%s' % i] = tmpdf.std(axis=1).values
    
    X=pd.DataFrame(X)
    
    if is_train:
        y=df[pd.date_range(date,periods=16)].values
        return X,y
    
    if columnname is not None:
        X.columns = ['%s_%s' % (columnname, c) for c in X.columns]

   # time.sleep(80)
    return X

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-08-01T14:40:51.804281Z","iopub.execute_input":"2024-08-01T14:40:51.804675Z","iopub.status.idle":"2024-08-01T14:40:52.500995Z","shell.execute_reply.started":"2024-08-01T14:40:51.804641Z","shell.execute_reply":"2024-08-01T14:40:52.499848Z"}}
val_x,val_y=prepare_dataset(store_family,store_family_promo,val_start,columnname='store_family')
#val.index=store_family.index
#val=val_2.reindex(store_family_index.set_index(['store_nbr','family']).index).reset_index(drop=True)


val_1=prepare_dataset(family,family_promo,val_start,columnname='family',is_train=False)
val_1.index=family.index
val_1=val_1.reindex(sales_train.index.get_level_values(1)).reset_index(drop=True)

validation=pd.concat([val_x,val_1,stores.reset_index()],axis=1).fillna(0)
del val_1,val_x

test_1=prepare_dataset(store_family,store_family_promo,test_start,columnname='store_family',is_train=False)
#test_2.index=store_family.index
#test_2=test_2.reindex(store_family_index.set_index(['store_nbr','family']).index).reset_index(drop=True)

test_2=prepare_dataset(family,family_promo,test_start,columnname='family',is_train=False)
test_2.index=family.index
test_2=test_2.reindex(sales_train.index.get_level_values(1)).reset_index(drop=True)


test=pd.concat([test_1,test_2,stores.reset_index()],axis=1).fillna(0)
del test_1,test_2

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-08-01T14:40:55.447776Z","iopub.execute_input":"2024-08-01T14:40:55.448428Z","iopub.status.idle":"2024-08-01T14:40:55.456688Z","shell.execute_reply.started":"2024-08-01T14:40:55.448392Z","shell.execute_reply":"2024-08-01T14:40:55.454902Z"}}
t = pd.to_datetime('2017-07-05')
train_start = []
for i in range(7):
    delta = pd.Timedelta(days=7 * i)
    train_start.append(t-delta)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-08-01T14:40:56.759102Z","iopub.execute_input":"2024-08-01T14:40:56.759464Z","iopub.status.idle":"2024-08-01T14:40:59.125221Z","shell.execute_reply.started":"2024-08-01T14:40:56.759439Z","shell.execute_reply":"2024-08-01T14:40:59.123898Z"}}
X_1=[]
y_1=[]
Z_1=[]
for start in train_start:
 
    train_x,train_y=prepare_dataset(store_family,store_family_promo,start,columnname='store_family')
   # Train.index=store_family.index
   # Train=train_2.reindex(store_family_index.set_index(['store_nbr','family']).index).reset_index(drop=True)
    
    
    train_1=prepare_dataset(family,family_promo,start,columnname='family',is_train=False)
    train_1.index=family.index
    train_1=train_1.reindex(sales_train.index.get_level_values(1)).reset_index(drop=True)
    
    train_all=pd.concat((train_x,train_1,stores.reset_index()),axis=1).fillna(0)
   # Verticaltrain_all=pd.concat((train_x,train_1)).fillna(0)

    X_1.append(train_all)
    y_1.append(train_y)


X_train=pd.concat(X_1,axis=0)
y_train=np.concatenate(y_1,axis=0)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-08-01T14:40:59.127067Z","iopub.execute_input":"2024-08-01T14:40:59.127416Z","iopub.status.idle":"2024-08-01T14:40:59.144911Z","shell.execute_reply.started":"2024-08-01T14:40:59.127385Z","shell.execute_reply":"2024-08-01T14:40:59.143515Z"}}
X_train= X_train[[i for i in X_train.columns if not 'proxy' in i]]
validation= validation[[i for i in validation.columns if not 'proxy' in i]]
test= test[[i for i in test.columns if not 'proxy' in i]]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-08-01T14:40:59.146581Z","iopub.execute_input":"2024-08-01T14:40:59.146973Z","iopub.status.idle":"2024-08-01T14:40:59.155791Z","shell.execute_reply.started":"2024-08-01T14:40:59.146942Z","shell.execute_reply":"2024-08-01T14:40:59.154684Z"}}
def rmsle_lgbm(y_pred, data):

    y_true = np.array(data.get_label())
    
    score = np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred), 2)))

    return 'rmsle', score, False

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-08-01T14:41:00.320215Z","iopub.execute_input":"2024-08-01T14:41:00.320613Z","iopub.status.idle":"2024-08-01T14:41:01.225059Z","shell.execute_reply.started":"2024-08-01T14:41:00.320580Z","shell.execute_reply":"2024-08-01T14:41:01.223860Z"}}
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T14:41:48.602296Z","iopub.execute_input":"2024-08-01T14:41:48.602685Z","iopub.status.idle":"2024-08-01T14:46:31.109701Z","shell.execute_reply.started":"2024-08-01T14:41:48.602654Z","shell.execute_reply":"2024-08-01T14:46:31.108449Z"}}
MAX_ROUNDS = 5000

val_pred = []
test_pred = []
cate_vars = []

params={
    'objective': 'regression',
    'learning_rate': 0.01,
    'metric': 'custom',
    'feature_fraction': 0.8,
    'min_data_in_leaf': 150,
     'n_jobs':-1,
    'subsample':0.6        
}

for i in range(16):
    print('step %d' %(i+1))

    
    dtrain=lgb.Dataset(
    X_train,label=y_train[:,i]
    )
    
    dval=lgb.Dataset(
    validation,label=val_y[:,i],reference=dtrain,
    )
    
    bst= lgb.train(
    params,dtrain,num_boost_round=MAX_ROUNDS,
    valid_sets=[dtrain,dval],
    callbacks=[lgb.early_stopping(150),lgb.log_evaluation(100)],
    #early_stopping_rounds=125,verbose_eval=100,
    feval=rmsle_lgbm)
   # **callbacks=[lgb.early_stopping(stopping_rounds=150), lgb.log_evaluation(150)]**)
    
    
   # print("\n".join(("%s: %.2f" % x) for x in sorted(
    #    zip(X_train.columns, bst.feature_importance("gain")),
     #   key=lambda x: x[1], reverse=True
    #)))
    

    
    val_pred.append(bst.predict(
    validation,num_iteration=bst.best_iteration or MAX_ROUNDS
    ))
    
    test_pred.append(bst.predict(
    test, num_iteration=bst.best_iteration or MAX_ROUNDS
    ))

# %% [code] {"execution":{"iopub.status.busy":"2024-07-25T13:06:11.455683Z","iopub.execute_input":"2024-07-25T13:06:11.456109Z","iopub.status.idle":"2024-07-25T13:06:11.497072Z","shell.execute_reply.started":"2024-07-25T13:06:11.456078Z","shell.execute_reply":"2024-07-25T13:06:11.495496Z"},"jupyter":{"outputs_hidden":false}}
q=rmsle_lgbm1(np.array(val_pred).transpose(),val_y)

if q[1]<z:
    print('improved')
    z=q[1]
elif q[1]>z:
    print("increased error ")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-08-01T14:41:46.889257Z","iopub.execute_input":"2024-08-01T14:41:46.889615Z","iopub.status.idle":"2024-08-01T14:41:46.894930Z","shell.execute_reply.started":"2024-08-01T14:41:46.889589Z","shell.execute_reply":"2024-08-01T14:41:46.893888Z"}}
def rmsle_lgbm1(y_pred, y_true):

   
    score = np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred), 2)))

    return 'rmsle', score, False

# %% [markdown]
# # # SUBMISSION

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-08-01T14:46:31.111717Z","iopub.execute_input":"2024-08-01T14:46:31.112162Z","iopub.status.idle":"2024-08-01T14:46:31.363295Z","shell.execute_reply.started":"2024-08-01T14:46:31.112121Z","shell.execute_reply":"2024-08-01T14:46:31.362276Z"}}
y_test=np.array(test_pred).transpose()
predictions=pd.DataFrame(
    y_test,index=sales_train.index,
    columns=pd.date_range("2017-08-16",periods=16)
).stack().to_frame("sales")

predictions.index.set_names(['store_nbr','family','date'],inplace=True)

df_test = pd.read_csv(r"/kaggle/input/store-sales-time-series-forecasting/test.csv"
                  ).set_index(['store_nbr', 'family', 'date'])

submission=df_test[['id']].join(predictions,how='left').fillna(0)
submission["sales"] = np.clip(np.expm1(submission["sales"]), 0, 1000)
submission.to_csv('lgb_sub.csv', float_format='%.4f', index=None)

# %% [code] {"jupyter":{"outputs_hidden":false}}
