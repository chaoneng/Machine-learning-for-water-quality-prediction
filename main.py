from flask import request,jsonify
import flask
import json
import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from collections import Counter
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,GRU ,Dense, Dropout
from keras.optimizers import *
from keras.callbacks import Callback
from keras.models import load_model
from sklearn.decomposition import PCA

app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.config["JSON_AS_ASCII"] = False     #設定顯示中文


@app.route('/training', methods=['GET','POST'])
def training():
    api_get_data_value=request.values
    api_get_data_json=request.json
    with open('./data/api_get_data_value_training.json', 'w', encoding='utf8') as outfile:
        json.dump(api_get_data_value, outfile, ensure_ascii=False)

    col_name=['device_id','device_no','device_name','obs_name','Time','ORP','DO','OXYGEN','PH','SALINITY','TEMP','h_24r','h_fx'
          ,'humd','ISA','pres','TAN','tmp','wdir','wdsd']
    stop_mse_values={'ORP':900,'OXYGEN':0.1,'PH':0.1,'SALINITY':0.1,'TEMP':0.1}
    train_data=list()
    time=list()
    target_name=list()
    TEST_SIZE=0.0

    with open('./data/api_get_data_value_training.json',encoding="utf-8") as f:
        data = json.load(f)
        for d in data:
            temp=d.replace("null", "0")
            new_data=eval(temp)    #dict
            if 'data' not in new_data:
                return "No data ,please input the data."
                break
            # print("訓練裝置名稱 : ",new_data['parameter'].get('t_name'),"資料開始時間 : ",new_data['parameter'].get('start_date'),
            # "資料結束時間 : ",new_data['parameter'].get('end_date'),"特徵 : ",new_data['parameter'].get('feature_list'),
            # "目標值 : ",new_data['parameter'].get('aims_list').keys(),"模型名稱 : ",new_data['parameter'].get('model'),
            # "測試集比例 : ",new_data['parameter'].get('test_set'))

            #目標值
            for i in new_data['parameter'].get('aims_list').keys():
                if i!='tan' and i!='ISA':
                    target_name.append(i)
            print("目標值 : ",target_name)

            model_name=new_data['parameter'].get('model')
            TEST_SIZE=float(new_data['parameter'].get('test_set'))/100  #test_size=0.4

            for i in new_data['data']['list']:
                for j in new_data['data']['list'][i]:
            #            print(j,data['data']['list'][i].get(j))
                    for k in new_data['data']['list'][i][j]:
                        temp=np.zeros(shape=(len(col_name)),dtype=object)
                        device_id=new_data['data']['list'][i].get('device_id')    #64
                        device_no=new_data['data']['list'][i].get('device_no')    #175
                        device_name=new_data['data']['list'][i].get('device_name')    #七股水試所標案-DE-1
                        obs_name=new_data['data']['list'][i].get('obs_name')    #七股水試所
                        if type(new_data['data']['list'][i][j]) != str:
                            time.append(k)
                            for h in new_data['data']['list'][i][j][k]:
                                temp[0]=device_id
                                temp[1]=device_no
                                temp[2]=device_name
                                temp[3]=obs_name
                                temp[col_name.index(h)]=new_data['data']['list'][i][j][k].get(h)
                            #     print(device_id,device_no,device_name,obs_name,k,h,new_data['data']['list'][i][j][k].get(h))
                            # print("..................................")
                            train_data.append(temp)


        def load_data(device_name,df):
            new_df=df[df['device_name']==device_name]
            new_df=new_df.sort_values(by = 'Time').reset_index(drop=True)       #按照時間先後排序資料

            #為零數值轉為眾數
            for i in range(5,new_df.shape[1]):
                if i!=11:
                    c = Counter(new_df.iloc[:,i].astype(float))
                    for j in c.most_common(2):
                        if j[0]!=0.0:
                            # print(i,j[0])
                            med=j[0]
                            new_df.iloc[:,i]=new_df.iloc[:,i].astype(float).replace(0.0,med)
            return new_df

        def load_model(model_type,inupt_size,stop_value):
            if model_type=='LSTM':
                early_stopping = EarlyStoppingByLossVal(monitor='val_mse', value=stop_value, verbose=1)
                model = Sequential()
                model.add(LSTM(25, input_length=72, input_dim=inupt_size))
                model.add(Dropout(0.2))
                model.add(Dense(1))
                model.compile(loss='mse',optimizer=RMSprop(lr=1e-2),metrics=['mse'])
                return model,early_stopping
            elif model_type=='GRU':
                early_stopping = EarlyStoppingByLossVal(monitor='val_mse', value=stop_value, verbose=1)
                model = Sequential()
                model.add(GRU(25, input_length=72, input_dim=inupt_size))
                model.add(Dropout(0.2))
                model.add(Dense(1))
                model.compile(loss='mse',optimizer=RMSprop(lr=1e-2),metrics=['mse'])
                return model,early_stopping

        class EarlyStoppingByLossVal(Callback):
            def __init__(self, monitor='val_loss', value=0.1, verbose=0):
                super(Callback, self).__init__()
                self.monitor = monitor
                self.value = value
                self.verbose = verbose

            def on_epoch_end(self, epoch, logs={}):
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

                if current < self.value:
                    if self.verbose > 0:
                        print("Epoch %05d: early stopping THR" % epoch)
                    self.model.stop_training = True

        def MSE(model,x_test,y_test):
            score=mean_squared_error(model.predict(x_test),y_test.astype(np.float))
            return score

         #Training
        def Training(device_name,df):
            # df.iloc[:,5:] = MinMaxScaler().fit_transform(df.iloc[:,5:])    #Set data range 0~1
            
            # LSTM
            for aims in target_name:
                if aims not in col_name:
                    aims=aims.upper()
                if aims=='DO' or aims=='do':
                    aims='OXYGEN'
                feature=df.iloc[:-12,5:].values
                target=df.loc[84:,aims].values
                new_feature=list()
                new_pca_feature=list()
                new_target=list()
                pca = PCA(n_components=4)
                pca_feature =pca.fit_transform(feature)
                # print(len(feature),len(target),aims)
                for i in range(len(feature)-72):
                    new_feature.append(feature[i:i+72])
                    new_pca_feature.append(pca_feature[i:i+72])
                    new_target.append(target[i])

                print("LSTM , ",aims,np.array(new_feature).shape,np.array(new_target).shape)

                # no PCA
                x_train, x_test, y_train, y_test = train_test_split(np.array(new_feature), np.array(new_target), test_size=TEST_SIZE, random_state=10)
                model,early_stopping=load_model('LSTM',12,0.1)    #stop_mse_values[aims]
                # callbacks=[early_stopping]
                history=model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=128,epochs=10,verbose=1)
                model.save('./model/NOPCA/'+str(device_name)+'_LSTM_'+str(aims)+'.h5')

                # PCA
                x_train, x_test, y_train, y_test = train_test_split(np.array(new_pca_feature), np.array(new_target), test_size=TEST_SIZE, random_state=10)
                model,early_stopping=load_model('LSTM',4,0.1)
                history=model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=128,epochs=10,verbose=1)
                model.save('./model/PCA/'+str(device_name)+'_LSTM_'+str(aims)+'_pca.h5')

            #GRU
            for aims in target_name:
                if aims not in col_name:
                    aims=aims.upper()
                if aims=='DO':
                    aims='OXYGEN'
                feature=df.iloc[:-12,5:].values
                target=df.loc[84:,aims].values
                new_feature=list()
                new_pca_feature=list()
                new_target=list()
                pca = PCA(n_components=4)
                pca_feature =pca.fit_transform(feature)
                for i in range(len(feature)-72):
                    new_feature.append(feature[i:i+72])
                    new_pca_feature.append(pca_feature[i:i+72])
                    new_target.append(target[i])

                print("GRU , ",aims,np.array(new_feature).shape,np.array(new_target).shape)

                # no PCA
                x_train, x_test, y_train, y_test = train_test_split(np.array(new_feature), np.array(new_target), test_size=TEST_SIZE, random_state=10)
                model,early_stopping=load_model('GRU',12,0.1)
                history=model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=128,epochs=10,verbose=1)
                model.save('./model/NOPCA/'+str(device_name)+'_GRU_'+str(aims)+'.h5')

                # PCA
                x_train, x_test, y_train, y_test = train_test_split(np.array(new_pca_feature), np.array(new_target), test_size=TEST_SIZE, random_state=10)
                model,early_stopping=load_model('GRU',4,0.1)
                history=model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=128,epochs=10,verbose=1)
                model.save('./model/PCA/'+str(device_name)+'_GRU_'+str(aims)+'_pca.h5')

            # XGBoost
            for aims in target_name:
                if aims not in col_name:
                    aims=aims.upper()
                if aims=='DO':
                    aims='OXYGEN'
                feature=df.iloc[:-12,5:].values
                target=df.loc[84:,aims].values
                new_feature=list()
                new_pca_feature=list()
                new_target=list()
                pca = PCA(n_components=4)
                pca_feature =pca.fit_transform(feature)
                for i in range(len(feature)-72):
                    new_feature.append(feature[i:i+72].reshape(864))
                    new_pca_feature.append(pca_feature[i:i+72].reshape(288))
                    new_target.append(target[i])

                print("XGBOOST , ",aims,np.array(new_feature).shape,np.array(new_pca_feature).shape,np.array(new_target).shape)

                # no PCA
                x_train, x_test, y_train, y_test = train_test_split(np.array(new_feature), np.array(new_target), test_size=TEST_SIZE, random_state=10)
                XGBR=xgb.XGBRegressor()
                XGBR.fit(x_train,y_train)
                joblib.dump(XGBR, './model/NOPCA/'+str(device_name)+'_XGBOOST_'+str(aims)+'.pkl') 
                print("XGBOOST MSE : ",MSE(XGBR,x_test,y_test))

                # PCA
                x_train, x_test, y_train, y_test = train_test_split(np.array(new_pca_feature), np.array(new_target), test_size=TEST_SIZE, random_state=10)
                XGBR=xgb.XGBRegressor()
                XGBR.fit(x_train,y_train)
                joblib.dump(XGBR, './model/PCA/'+str(device_name)+'_XGBOOST_'+str(aims)+'_pca.pkl') 
                print("XGBOOST MSE : ",MSE(XGBR,x_test,y_test))


        df=pd.DataFrame(data=train_data,columns=col_name)
        df['Time']=time
        df.to_csv('./data/鹿港水試所_2020.csv',index=False,encoding='utf-8-sig')
        # for i in set(df['device_name']):
        #     new_df=load_data(i,df)
        #     print(i,new_df.shape)
        #     Training(i,new_df)
            
    return "Training completed"



@app.route('/prediction', methods=['GET','POST'])
def prediction():

    col_name=['device_id','device_no','device_name','obs_name','Time','ORP','OXYGEN','PH','SALINITY','TEMP','h_24r','h_fx','humd','pres','tmp','wdir','wdsd']
    train_data=list()
    time=list()
    target_name=list()

    api_get_data_value=request.values
    api_get_data_json=request.json
    with open('./data/api_get_data_value.json', 'w', encoding='utf8') as outfile:
        json.dump(api_get_data_value, outfile, ensure_ascii=False)

    with open('./data/api_get_data_value.json',encoding="utf-8") as f:
        data = json.load(f)
        for d in data:
            temp=d.replace("null", "0")
            new_data=eval(temp)    #dict
            if 'data' not in new_data:
                return "No data ,please input the data."
                break

            model_name=new_data['parameter'].get('model')
            #目標值
            for i in new_data['parameter'].get('aims_list').keys():
                if i!='tan' and i!='ISA':
                    if i not in col_name:
                        i=i.upper()
                        target_name.append(i)
                    if i =='do':
                        target_name.append('OXYGEN')
                    else :
                        target_name.append(i)
            print("目標值 : ",target_name)

            for i in new_data['data']['list']:
                for j in new_data['data']['list'][i]:
                    for k in new_data['data']['list'][i][j]:
                        temp=np.zeros(shape=(len(col_name)),dtype=object)
                        device_id=new_data['data']['list'][i].get('device_id')    #64
                        device_no=new_data['data']['list'][i].get('device_no')    #175
                        device_name=new_data['data']['list'][i].get('device_name')    #七股水試所標案-DE-1
                        obs_name=new_data['data']['list'][i].get('obs_name')    #七股水試所
                        if type(new_data['data']['list'][i][j]) != str:
                            time.append(k)
                            for h in new_data['data']['list'][i][j][k]:
                                temp[0]=device_id
                                temp[1]=device_no
                                temp[2]=device_name
                                temp[3]=obs_name
                                temp[col_name.index(h)]=new_data['data']['list'][i][j][k].get(h)
                            #     print(device_id,device_no,device_name,obs_name,k,h,new_data['data']['list'][i][j][k].get(h))
                            # print("..................................")
                            train_data.append(temp)

        def load_data(device_name,df):
            new_df=df[df['device_name']==device_name]
            raw_test_df=new_df.sort_values(by = 'Time').reset_index(drop=True)       #按照時間先後排序資料
            test_df=new_df.sort_values(by = 'Time').reset_index(drop=True)       #按照時間先後排序資料

            #為零數值轉為眾數
            for i in range(5,test_df.shape[1]):
                if i!=11:
                    c = Counter(test_df.iloc[:,i].astype(float))
                    for j in c.most_common(2):
                        if j[0]!=0.0:
                            # print(i,j[0])
                            med=j[0]
                            test_df.iloc[:,i]=test_df.iloc[:,i].astype(float).replace(0.0,med)
            return test_df,raw_test_df

        df=pd.DataFrame(data=train_data,columns=col_name)
        df['Time']=time
        # minmax=pd.to_numeric(df.iloc[:,7], downcast="float")
        # MIN=minmax.min()
        # MAX=minmax.max()
        # scale=MinMaxScaler()
        # df.iloc[:,5:] = scale.fit_transform(df.iloc[:,5:])
        field=set(df['device_name'])
        new_field=['七股水試所標案-DE-2']
        # for i in field:
        #     new_field.append(i)
        parsed=[]
        lines=0
        pre_df=pd.DataFrame()
        # pre_df=pd.DataFrame(columns=['device_name','model','ui_time','raw_'+str(j),'nopca_predict_'+str(j),'pca_predict_'+str(j),'predict_time'])
        for f in new_field:
            test_df,raw_test_df=load_data(f,df)
            for j in target_name:
                feature=test_df.iloc[:-12,5:].values
                predict_time=test_df.iloc[84:,4].values
                ui_time=test_df.iloc[72:-12,4].values
                new_feature=list()
                new_pca_feature=list()
                pca = PCA(n_components=4)
                pca_feature =pca.fit_transform(feature)
                
                # LSTM
                if model_name=='LSTM':
                    for k in range(len(feature)-72):
                        new_feature.append(feature[k:k+72])
                        new_pca_feature.append(pca_feature[k:k+72])

                    print('./model/NOPCA/'+str(f)+'_LSTM_'+str(j)+'.h5')
                    model1=load_model('./model/NOPCA/'+str(f)+'_LSTM_'+str(j)+'.h5')
                    pre=model1.predict(np.array(new_feature),batch_size=128)
                    model2=load_model('./model/PCA/'+str(f)+str('_LSTM_')+str(j)+str('_pca.h5'))
                    pre1=model2.predict(np.array(new_pca_feature),batch_size=128)
                    new_lines=len(pre)
                    # print(pre,pre1)
                    # if lines==0:
                    pre_df['raw_'+str(j)]=raw_test_df.loc[84:,j]
                    pre_df['device_name']=f
                    pre_df['model']='LSTM'
                    pre_df['nopca_predict_'+str(j)]=pre
                    pre_df['pca_predict_'+str(j)]=pre1
                    pre_df['predict_time']=predict_time
                    pre_df['ui_time']=ui_time
                        
                        # pre_df.to_csv('./predict/'+str(j)+'/'+str(f)+'_LSTM_prediction_nopca.csv',encoding='utf-8-sig',index=False)
                    # else:
                    #     pre_df['raw_'+str(j)][lines:lines+new_lines]=raw_test_df.loc[84:,j]
                    #     pre_df['device_name'][lines:lines+new_lines]=f
                    #     pre_df['nopca_predict_'+str(j)][lines:lines+new_lines]=pre
                    #     pre_df['pca_predict_'+str(j)][lines:lines+new_lines]=pre1
                    #     pre_df['predict_time'][lines:lines+new_lines]=predict_time
                    #     pre_df['model'][lines:lines+new_lines]='LSTM'
                    #     pre_df['ui_time'][lines:lines+new_lines]=ui_time


                #GRU
                elif model_name=='GRU':
                    for k in range(len(feature)-72):
                        new_feature.append(feature[k:k+72])
                        new_pca_feature.append(pca_feature[k:k+72])

                    print('./model/NOPCA/'+str(f)+'_GRU_'+str(j)+'.h5')
                    model3=load_model('./model/NOPCA/'+str(f)+'_GRU_'+str(j)+'.h5')
                    pre=model3.predict(np.array(new_feature),batch_size=128)
                    model4=load_model('./model/PCA/'+str(f)+str('_GRU_')+str(j)+str('_pca.h5'))
                    pre1=model4.predict(np.array(new_pca_feature),batch_size=128)
                    new_lines=len(pre)
                    # print(pre,pre1)
                    # if lines==0:
                    pre_df['raw_'+str(j)]=raw_test_df.loc[84:,j]
                    pre_df['device_name']=f
                    pre_df['model']='GRU'
                    pre_df['nopca_predict_'+str(j)]=pre
                    pre_df['pca_predict_'+str(j)]=pre1
                    pre_df['predict_time']=predict_time
                    pre_df['ui_time']=ui_time
                        
                   

                #XGBOOST
                elif model_name=='XGBOOST':
                    for i in range(len(feature)-72):
                        new_feature.append(feature[i:i+72].reshape(864))
                        new_pca_feature.append(pca_feature[i:i+72].reshape(288))

                    # pre_df=pd.DataFrame(columns=['device_name','model','ui_time','raw_'+str(j),'pca_predict_'+str(j),'predict_time'])
                    print('./model/NOPCA/'+str(f)+'_XGBOOST_'+str(j)+'.h5')
                    model5=joblib.load('./model/NOPCA/'+str(f)+str('_XGBOOST_')+str(j)+str('.pkl'))
                    pre=model5.predict(np.array(new_feature))
                    model6=joblib.load('./model/PCA/'+str(f)+str('_XGBOOST_')+str(j)+str('_pca.pkl'))
                    pre1=model6.predict(np.array(new_pca_feature))
                    new_lines=len(pre)
                    # print(pre,pre1)
                    # if lines==0:
                    pre_df['raw_'+str(j)]=raw_test_df.loc[84:,j]
                    pre_df['device_name']=f
                    pre_df['model']='XGBOOST'
                    pre_df['nopca_predict_'+str(j)]=pre
                    pre_df['pca_predict_'+str(j)]=pre1
                    pre_df['predict_time']=predict_time
                    pre_df['ui_time']=ui_time
                        
                    
            # lines+=new_lines
        # pre_df.to_csv('./predict/prediction_pca.csv',encoding='utf-8-sig',index=False)
        # result = pre_df.to_json(orient="records")
        # parsed = json.loads(result)
        # with open('test.json', 'w') as outfile:       #"w" 以寫方式開啟，只能寫檔案， 如果檔案不存在，建立該檔案，如果檔案已存在，先清空，再開啟檔案
        #     json.dump(parsed, outfile,ensure_ascii=False)
        return jsonify(pre_df.to_dict(orient='records'))

    # x=[[1,2,3],[4,5,6],[7,8,9]]
    # scale=MinMaxScaler()
    # x1=scale.fit_transform(x)
    # x2=scale.inverse_transform(x1)
    # test=[
    #     {
    #         "field": "岡山",
    #         "model": "XGBoost",
    #         "接收時間": "2020-09-09 10:40:00",
    #         "溶氧量DO(ppm)": 3.77
    #     }   
    # ]

    return "Prediction completed"


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    # run_simple(hostname ='0.0.0.0',use_reloader = True, port=5000,application=app, processes=1)
    app.run(debug=True, host='0.0.0.0', port=5000, processes=1)
    