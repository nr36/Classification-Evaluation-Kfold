import pandas as pd
def overallOutput():
    
    import numpy as np
    import tensorflow as tf
    from tensorflow.python.ops.math_ops import reduce_prod
    import warnings
    warnings.filterwarnings("ignore")
    import math
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    try:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except Exception as e:
        print('')
    #print('tensorflow version : ',tf. __version__)
    #print('numpy version : ', np. __version__)
    data=pd.read_csv('wine.csv',header=None)
    df=data.sample(frac=1)
    #print(df.head)
    # First column represents the quality of wine(1 or 2) so it is selected as label
    labels=df.iloc[:,0]
    features=df.iloc[:,1:14]
    X=features
    y=labels
    TP_NB=[]
    FP_NB=[]
    FN_NB=[]
    TN_NB=[]
    TPR_NB=[]
    FPR_NB=[]
    TNR_NB=[]
    FNR_NB=[]
    RECALL_NB=[]
    PRECISION_NB=[]
    F1_SCORE_NB=[]
    ACCURACY_NB=[]
    ERROR_RATE_NB=[]
    BACC_NB=[]
    TSS_NB=[]
    HSS_NB=[]
    BS_NB=[]
    BSS_NB=[]
    
    TP_RF=[]
    FP_RF=[]
    FN_RF=[]
    TN_RF=[]
    TPR_RF=[]
    FPR_RF=[]
    TNR_RF=[]
    FNR_RF=[]
    RECALL_RF=[]
    PRECISION_RF=[]
    F1_SCORE_RF=[]
    ACCURACY_RF=[]
    ERROR_RATE_RF=[]
    BACC_RF=[]
    TSS_RF=[]
    HSS_RF=[]
    BS_RF=[]
    BSS_RF=[]
    
    TP_LSTM=[]
    FP_LSTM=[]
    FN_LSTM=[]
    TN_LSTM=[]
    TPR_LSTM=[]
    FPR_LSTM=[]
    TNR_LSTM=[]
    FNR_LSTM=[]
    RECALL_LSTM=[]
    PRECISION_LSTM=[]
    F1_SCORE_LSTM=[]
    ACCURACY_LSTM=[]
    ERROR_RATE_LSTM=[]
    BACC_LSTM=[]
    TSS_LSTM=[]
    HSS_LSTM=[]
    BS_LSTM=[]
    BSS_LSTM=[]
    
    def evaluation_metricsNB(TP,TN,FP,FN):
        
        TP=TP
        TN=TN
        FP=FP
        FN=FN
        TPR=TP/(TP+FN)
        TNR=TN/(TN+FP)
        FPR=FP/(FP+TN)
        FNR=FN/(FN+TP)
        RECALL=TPR
        PRECISION=TP/(TP+FP)
        F1_SCORE=(2*TP)/(2*TP+FP+FN)
        ACCURACY=(TP+TN)/(TP+FP+TN+FN)
        ERROR_RATE=1-ACCURACY
        BACC=(TPR+TNR)/2
        TSS=TPR-FPR
        HSS=2*(TP*TN-FP*FN)/(((TP+FN)*(FN+TN))+((TP+FP)*(FP+TN)))
        sum_y=0
        for n in range(len(y_test)):
            sum_y+=(y_test[n]-y_predNB[n])**2 
        BS=sum_y/len(y_test)
        
        y_meantemp=0
        for i in range(len(y_test)):
            y_meantemp+=y_test[i]
        ymean=y_meantemp/len(y_test)
        #BSS
        temp=0
        for i in range(len(y_test)):
            temp+=(y_test[i]-ymean)**2
        temp=temp/len(y_test)
        BSS=BS/temp
    
        TP_NB.append(TP)
        FP_NB.append(FP)
        FN_NB.append(FN)
        TN_NB.append(TN)
        TPR_NB.append(TPR)
        FPR_NB.append(FPR)
        TNR_NB.append(TNR)
        FNR_NB.append(FNR)
        RECALL_NB.append(RECALL)
        PRECISION_NB.append(PRECISION)
        F1_SCORE_NB.append(F1_SCORE)
        ACCURACY_NB.append(ACCURACY)
        ERROR_RATE_NB.append(ERROR_RATE)
        BACC_NB.append(BACC)
        TSS_NB.append(TSS)
        HSS_NB.append(HSS)
        BS_NB.append(BS)
        BSS_NB.append(BSS)
        
    def evaluation_metricsRF(TP,TN,FP,FN):
        
        TP=TP
        TN=TN
        FP=FP
        FN=FN
        TPR=TP/(TP+FN)
        TNR=TN/(TN+FP)
        FPR=FP/(FP+TN)
        FNR=FN/(FN+TP)
        RECALL=TPR
        PRECISION=TP/(TP+FP)
        F1_SCORE=(2*TP)/(2*TP+FP+FN)
        ACCURACY=(TP+TN)/(TP+FP+TN+FN)
        ERROR_RATE=1-ACCURACY
        BACC=(TPR+TNR)/2
        TSS=TPR-FPR
        HSS=2*(TP*TN-FP*FN)/(((TP+FN)*(FN+TN))+((TP+FP)*(FP+TN)))
        sum_y=0
        for n in range(len(y_test)):
            sum_y+=(y_test[n]-y_predRF[n])**2 
        BS=sum_y/len(y_test)
        
        y_meantemp=0
        for i in range(len(y_test)):
            y_meantemp+=y_test[i]
        ymean=y_meantemp/len(y_test)
        
        temp=0
        for i in range(len(y_test)):
            temp+=(y_test[i]-ymean)**2
        temp=temp/len(y_test)
        BSS=BS/temp
    
        TP_RF.append(TP)
        FP_RF.append(FP)
        FN_RF.append(FN)
        TN_RF.append(TN)
        TPR_RF.append(TPR)
        FPR_RF.append(FPR)
        TNR_RF.append(TNR)
        FNR_RF.append(FNR)
        RECALL_RF.append(RECALL)
        PRECISION_RF.append(PRECISION)
        F1_SCORE_RF.append(F1_SCORE)
        ACCURACY_RF.append(ACCURACY)
        ERROR_RATE_RF.append(ERROR_RATE)
        BACC_RF.append(BACC)
        TSS_RF.append(TSS)
        HSS_RF.append(HSS)
        BS_RF.append(BS)
        BSS_RF.append(BSS)
        
    def evaluation_metrics_lstm(TP,TN,FP,FN):
        
        TP=TP
        TN=TN
        FP=FP
        FN=FN
        TPR=TP/(TP+FN)
        TNR=TN/(TN+FP)
        FPR=FP/(FP+TN)
        FNR=FN/(FN+TP)
        RECALL=TPR
        PRECISION=TP/(TP+FP)
        if math.isnan(PRECISION):
            PRECISION_LSTM.append(np.nan) 
            
        F1_SCORE=(2*TP)/(2*TP+FP+FN)
        ACCURACY=(TP+TN)/(TP+FP+TN+FN)
        ERROR_RATE=1-ACCURACY
        BACC=(TPR+TNR)/2
        TSS=TPR-FPR
        HSS=2*(TP*TN-FP*FN)/(((TP+FN)*(FN+TN))+((TP+FP)*(FP+TN)))
        sum_y=0
        for n in range(len(y_test)):
            sum_y+=(y_test[n]-y_predLSTM[n])**2 
        BS=sum_y/len(y_test)
        
        y_meantemp=0
        for i in range(len(y_test)):
            y_meantemp+=y_test[i]
        ymean=y_meantemp/len(y_test)
    
        temp=0
        for i in range(len(y_test)):
            temp+=(y_test[i]-ymean)**2
        temp=temp/len(y_test)
        BSS=BS/temp
    
        TP_LSTM.append(TP)
        FP_LSTM.append(FP)
        FN_LSTM.append(FN)
        TN_LSTM.append(TN)
        TPR_LSTM.append(TPR)
        FPR_LSTM.append(FPR)
        TNR_LSTM.append(TNR)
        FNR_LSTM.append(FNR)
        RECALL_LSTM.append(RECALL)
        
        F1_SCORE_LSTM.append(F1_SCORE)
        ACCURACY_LSTM.append(ACCURACY)
        ERROR_RATE_LSTM.append(ERROR_RATE)
        BACC_LSTM.append(BACC)
        TSS_LSTM.append(TSS)
        HSS_LSTM.append(HSS)
        BS_LSTM.append(BS)
        BSS_LSTM.append(BSS)
    
    
    X=np.array(X)
    y=np.array(y)
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix 
    
    
    kf = KFold(n_splits=10)
    
    TN_TOTALNB=0
    TP_TOTALNB=0
    FP_TOTALNB=0
    FN_TOTALNB=0
    
    TN_TOTALRF=0
    TP_TOTALRF=0
    FP_TOTALRF=0
    FN_TOTALRF=0
    
    TN_TOTAL_LSTM=0
    TP_TOTAL_LSTM=0
    FP_TOTAL_LSTM=0
    FN_TOTAL_LSTM=0
    
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index] 
        
    ###########################      Model1 Naive Bayes     ###################################
    
        modelNB = GaussianNB()                       
        modelNB.fit(X_train, y_train) 
        y_predNB = modelNB.predict(X_test)
        cnf_matrixNB = confusion_matrix(y_test, y_predNB) 
        [[TNNB, FPNB],
        [FNNB, TPNB]]=cnf_matrixNB
        
        evaluation_metricsNB(TPNB,TNNB,FPNB,FNNB)
        
        TN_TOTALNB+=TNNB
        TP_TOTALNB+=TPNB
        FP_TOTALNB+=FPNB
        FN_TOTALNB+=FNNB
        
    ###########################      Model2 Random Forest     ###################################
        
        rf= RandomForestClassifier(n_estimators=20, random_state=0)
        rf.fit(X_train, y_train)
        y_predRF=rf.predict(X_test)
        cnf_matrixRF = confusion_matrix(y_test, y_predRF)
        [[TNRF, FPRF],
        [FNRF, TPRF]]=cnf_matrixRF
        
        evaluation_metricsRF(TPRF,TNRF,FPRF,FNRF)
        
        TN_TOTALRF+=TNRF
        TP_TOTALRF+=TPRF
        FP_TOTALRF+=FPRF
        FN_TOTALRF+=FNRF
        
    ###########################      Model 3 LSTM       #########################################
    
      
    #    Reshape the data to match 3 dimension for LSTM layers.
    
        X_train1 = X_train.reshape(X_train.shape[0], X_train.shape[1],1)
        X_test1 = X_test.reshape(X_test.shape[0], X_test.shape[1],1)
        
    #     print('X_train.shape:', X_train.shape)
    #     print('y_train.shape:', y_train.shape)
    #     print('X_test.shape:', X_test.shape)
    #     print('y_test.shape:', y_test.shape)
    
        lstm_model = tf.keras.Sequential()
        lstm_model.add(tf.keras.layers.LSTM(64,return_sequences=True, return_state=False,input_shape=(X_test1.shape[1],X_test1.shape[2])))
        lstm_model.add(tf.keras.layers.LSTM(64, return_sequences=True, return_state=False))
        lstm_model.add(tf.keras.layers.LSTM(64, return_sequences=True, return_state=False))
        lstm_model.add(tf.keras.layers.Flatten())
        lstm_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
        # Compile the Model
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        lstm_model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
        #lstm_model.summary()
        lstm_model.fit(X_train1, y_train,batch_size=1, verbose = 0)
        y_predLSTM = lstm_model.predict(X_test1)
        score = lstm_model.evaluate(X_test1, y_test,verbose=0)
        cnf_matrix_LSTM = confusion_matrix(y_test, y_predLSTM)
        [[TNlstm, FPlstm],
        [FNlstm, TPlstm]]=cnf_matrix_LSTM
        
        evaluation_metrics_lstm(TPlstm,TNlstm,FPlstm,FNlstm)
        
        TN_TOTAL_LSTM+=TNlstm
        TP_TOTAL_LSTM+=TPlstm
        FP_TOTAL_LSTM+=FPlstm
        FN_TOTAL_LSTM+=FNlstm
    
    dfa=pd.DataFrame({
                    "TP": TP_NB,
                    "FP": FP_NB,
                    "FN": FN_NB,
                    "TN": TN_NB,
                    "TPR":TPR_NB,
                    "FPR":FPR_NB,
                    "TNR":TNR_NB,
                    "FNR":FNR_NB,
                    "RECALL":RECALL_NB,
                    'PRECISION':PRECISION_NB,
                    'F1_SCORE':F1_SCORE_NB,
                    'Accuracy':ACCURACY_NB,
                    'Error rate':ERROR_RATE_NB,
                    'BACC':BACC_NB,
                    'TSS':TSS_NB,
                    'HSS':HSS_NB,
                    'BS' :BS_NB,
                    'BSS':BSS_NB},
                    index=['Naive-Bayes','Naive-Bayes','Naive-Bayes','Naive-Bayes','Naive-Bayes',
                           'Naive-Bayes','Naive-Bayes','Naive-Bayes','Naive-Bayes','Naive-Bayes',])
    dfb=pd.DataFrame({
                    "TP": TP_RF,
                    "FP": FP_RF,
                    "FN": FN_RF,
                    "TN": TN_RF,
                    "TPR":TPR_RF,
                    "FPR":FPR_RF,
                    "TNR":TNR_RF,
                    "FNR":FNR_RF,
                    "RECALL":RECALL_RF,
                    'PRECISION':PRECISION_RF,
                    'F1_SCORE':F1_SCORE_RF,
                    'Accuracy':ACCURACY_RF,
                    'Error rate':ERROR_RATE_RF,
                    'BACC':BACC_RF,
                    'TSS':TSS_RF,
                    'HSS':HSS_RF,
                    'BS' :BS_RF,
                    'BSS':BSS_RF},
                    index=['Random-Forest','Random-Forest','Random-Forest','Random-Forest','Random-Forest',
                          'Random-Forest','Random-Forest','Random-Forest','Random-Forest','Random-Forest',])
    
    dfc=pd.DataFrame({
                    "TP": TP_LSTM,
                    "FP": FP_LSTM,
                    "FN": FN_LSTM,
                    "TN": TN_LSTM,
                    "TPR":TPR_LSTM,
                    "FPR":FPR_LSTM,
                    "TNR":TNR_LSTM,
                    "FNR":FNR_LSTM,
                    "RECALL":RECALL_LSTM,
                    'PRECISION':PRECISION_LSTM,
                    'F1_SCORE':F1_SCORE_LSTM,
                    'Accuracy':ACCURACY_LSTM,
                    'Error rate':ERROR_RATE_LSTM,
                    'BACC':BACC_LSTM,
                    'TSS':TSS_LSTM,
                    'HSS':HSS_LSTM,
                    'BS' :BS_LSTM,
                    'BSS':BSS_LSTM},
                    index=['LSTM','LSTM','LSTM','LSTM','LSTM','LSTM',
                          'LSTM','LSTM','LSTM','LSTM',])
    
    d1=pd.concat([dfa.iloc[0:1],dfb.iloc[0:1],dfc.iloc[0:1]])
    d2=pd.concat([dfa.iloc[1:2],dfb.iloc[1:2],dfc.iloc[1:2]])
    d3=pd.concat([dfa.iloc[2:3],dfb.iloc[2:3],dfc.iloc[2:3]])
    d4=pd.concat([dfa.iloc[3:4],dfb.iloc[3:4],dfc.iloc[3:4]])
    d5=pd.concat([dfa.iloc[4:5],dfb.iloc[4:5],dfc.iloc[4:5]])
    d6=pd.concat([dfa.iloc[5:6],dfb.iloc[5:6],dfc.iloc[5:6]])
    d7=pd.concat([dfa.iloc[6:7],dfb.iloc[6:7],dfc.iloc[6:7]])
    d8=pd.concat([dfa.iloc[7:8],dfb.iloc[7:8],dfc.iloc[7:8]])
    d9=pd.concat([dfa.iloc[8:9],dfb.iloc[8:9],dfc.iloc[8:9]])
    d10=pd.concat([dfa.iloc[9:10],dfb.iloc[9:10],dfc.iloc[9:10]])
    
    dfEachFold=pd.concat([d1,d2,d3,d4,d5,d6,d7,d8,d9,d10],keys=('KFOLD-1','KFOLD-2','KFOLD-3','KFOLD-4','KFOLD-5','KFOLD-6','KFOLD-7',
                     'KFOLD-8','KFOLD-9','KFOLD-10'))
        
    #print(dfEachFold)
    
    # Aggregating for Naive Bayes
    
    TN=TN_TOTALNB/10
    TP=TP_TOTALNB/10
    FN=FN_TOTALNB/10
    FP=FP_TOTALNB/10
    
    TPR=TP/(TP+FN)
    TNR=TN/(TN+FP)
    FPR=FP/(FP+TN)
    FNR=FN/(FN+TP)
    RECALL=TPR
    PRECISION=TP/(TP+FP)
    F1_SCORE=(2*TP)/(2*TP+FP+FN)
    ACCURACY=(TP+TN)/(TP+FP+TN+FN)
    ERROR_RATE=1-ACCURACY
    BACC=(TPR+TNR)/2
    TSS=TPR-FPR
    HSS=2*(TP*TN-FP*FN)/(((TP+FN)*(FN+TN))+((TP+FP)*(FP+TN)))
    sum_y=0
    for n in range(len(y_test)):
        sum_y+=(y_test[n]-y_predNB[n])**2 
    BS=sum_y/len(y_test)
        
    y_meantemp=0
    for i in range(len(y_test)):
        y_meantemp+=y_test[i]
    ymean=y_meantemp/len(y_test)
    temp=0
    for i in range(len(y_test)):
        temp+=(y_test[i]-ymean)**2
    temp=temp/len(y_test)
    BSS=BS/temp
    
    dfavg1=pd.DataFrame({"TP": TP,
                    "FP": FP,
                    "FN": FN,
                    "TN": TN,
                    "TPR":TPR,
                    "FPR":FPR,
                    "TNR":TNR,
                    "FNR":FNR,
                    "RECALL":RECALL,
                    'PRECISION':PRECISION,
                    'F1_SCORE':F1_SCORE,
                    'Accuracy':ACCURACY,
                    'Error rate':ERROR_RATE,
                    'BACC':BACC,
                    'TSS':TSS,
                    'HSS':HSS,
                    'BS' :BS,
                    'BSS':BSS
                     },
                     index=["NAIVE BAYES"])
    
    
    #Averaging for random forest model
    
    TN=TN_TOTALRF/10
    TP=TP_TOTALRF/10
    FP=FP_TOTALRF/10
    FN=FN_TOTALRF/10
    
    TPR=TP/(TP+FN)
    TNR=TN/(TN+FP)
    FPR=FP/(FP+TN)
    FNR=FN/(FN+TP)
    RECALL=TPR
    PRECISION=TP/(TP+FP)
    F1_SCORE=(2*TP)/(2*TP+FP+FN)
    ACCURACY=(TP+TN)/(TP+FP+TN+FN)
    ERROR_RATE=1-ACCURACY
    BACC=(TPR+TNR)/2
    TSS=TPR-FPR
    HSS=2*(TP*TN-FP*FN)/(((TP+FN)*(FN+TN))+((TP+FP)*(FP+TN)))
    sum_y=0
    for n in range(len(y_test)):
        sum_y+=(y_test[n]-y_predRF[n])**2 
    BS=sum_y/len(y_test)
        
    y_meantemp=0
    for i in range(len(y_test)):
        y_meantemp+=y_test[i]
    ymean=y_meantemp/len(y_test)
    temp=0
    for i in range(len(y_test)):
        temp+=(y_test[i]-ymean)**2
    temp=temp/len(y_test)
    BSS=BS/temp
    
    dfavg2=pd.DataFrame({"TP": TP,
                    "FP": FP,
                    "FN": FN,
                    "TN": TN,
                    "TPR":TPR,
                    "FPR":FPR,
                    "TNR":TNR,
                    "FNR":FNR,
                    "RECALL":RECALL,
                    'PRECISION':PRECISION,
                    'F1_SCORE':F1_SCORE,
                    'Accuracy':ACCURACY,
                    'Error rate':ERROR_RATE,
                    'BACC':BACC,
                    'TSS':TSS,
                    'HSS':HSS,
                    'BS' :BS,
                    'BSS':BSS
                     },
                     index=["RANDOM FOREST"])
    
    # Aggregating for LSTM model
    
    TN=TN_TOTAL_LSTM/10
    TP=TP_TOTAL_LSTM/10
    FP=FP_TOTAL_LSTM/10
    FN=FN_TOTAL_LSTM/10
    
    TPR=TP/(TP+FN)
    TNR=TN/(TN+FP)
    FPR=FP/(FP+TN)
    FNR=FN/(FN+TP)
    RECALL=TPR
    PRECISION=TP/(TP+FP)
    F1_SCORE=(2*TP)/(2*TP+FP+FN)
    ACCURACY=(TP+TN)/(TP+FP+TN+FN)
    ERROR_RATE=1-ACCURACY
    BACC=(TPR+TNR)/2
    TSS=TPR-FPR
    HSS=2*(TP*TN-FP*FN)/(((TP+FN)*(FN+TN))+((TP+FP)*(FP+TN)))
    sum_y=0
    for n in range(len(y_test)):
        sum_y+=(y_test[n]-y_predLSTM[n])**2 
    BS=sum_y/len(y_test)
        
    y_meantemp=0
    for i in range(len(y_test)):
        y_meantemp+=y_test[i]
    ymean=y_meantemp/len(y_test)
    temp=0
    for i in range(len(y_test)):
        temp+=(y_test[i]-ymean)**2
    temp=temp/len(y_test)
    BSS=BS/temp
    
    dfavg3=pd.DataFrame({"TP": TP,
                    "FP": FP,
                    "FN": FN,
                    "TN": TN,
                    "TPR":TPR,
                    "FPR":FPR,
                    "TNR":TNR,
                    "FNR":FNR,
                    "RECALL":RECALL,
                    'PRECISION':PRECISION,
                    'F1_SCORE':F1_SCORE,
                    'Accuracy':ACCURACY,
                    'Error rate':ERROR_RATE,
                    'BACC':BACC,
                    'TSS':TSS,
                    'HSS':HSS,
                    'BS' :BS,
                    'BSS':BSS
                     },
                     index=["LSTM"])
    
    df_avg = pd.concat([dfavg1, dfavg2,dfavg3])
    #print(df_avg)
    
    import openpyxl
    import xlsxwriter
    import xlwt
    writer = pd.ExcelWriter('FinalResult.xlsx', engine='xlsxwriter')
    
    #write each DataFrame to a specific sheet
    dfEachFold.to_excel(writer, sheet_name='EachFold')
    df_avg.to_excel(writer, sheet_name='Overall')
    
    #close the Pandas Excel writer and output the Excel file
    writer.save()
    return dfEachFold, df_avg