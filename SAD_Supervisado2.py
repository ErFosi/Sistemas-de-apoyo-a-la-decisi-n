# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import getopt
import csv
import os
import sys
import numpy as np
import pandas
import pandas as pd
import sklearn as sk
import imblearn
from _cffi_backend import typeof
from fs.time import datetime_to_epoch
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle


# Press the green button in the gutter to run the script.


def KNN():
    NumTest=0
    k = 1
    d = 1
    path = './'
    #f = "trainHalfHalf"
    #target = sys.argv[3]
    #f=str(sys.argv[4])
    oFile = "output.out"
    hayTrain=False
    balanced=False
    nproc=False


    print('ARGV   :',sys.argv[1:])
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'o:k:d:p:f:h:opt:alg:t:test:b:np:s',['output=','k=','d=','path=','iFile','h','opt=','alg=','t=','test=','b=','np=','s='])
    except getopt.GetoptError as err:
        print('ERROR:',err)
        sys.exit(1)
    #print('OPTIONS   :',options)
    ListOPT=False
    alg=""
    s=False
    for opt,arg in options:
        if opt in ('-o','--output'):
            oFile = arg

        elif opt == '-k':
            K = arg
        elif opt== '-s':
            s=True
        elif opt in ('--np'):
            nproc = True
        elif opt in ('--alg'):
            alg = arg
        elif opt ==  '-d':
            P = arg
        elif opt ==  '-b':
            balanced=True
        elif opt ==  '-t':
            target = arg
        elif opt in ('-p', '--path'):
            p = arg
        elif opt in ('-f', '--file'):
            f = arg
            print(f)
        elif opt in ('-h','--help'):
            print(' -o outputFile \n -k range of k(x,y..) \n -d range of d(x,y...) \n -p inputFilePath \n -f inputFileName \n -t targetColumn\n -alg KNN or DTREE \n -opt x,y,z\n --test test file name\n -b balance the features\n -np no process \n')
            print("Options are used for a default value for every column, after executing check config.csv out")
            print("Example:python3 SAD_Supervisado.py --opt 1,1,1 -k 1,3,5 -d 1,2 --alg KNN -f pruebaexamen.csv -p ./ -t Class -np 1 --test test2.csv ")
            print("Using the option --opt 1,1,1 and -s will create an option file with the provided options, not using any of those 2 options will not create config.csv and will try to read it, using --opt without -s will create and use it for the preprocessing and training")
            print("using --test 'file' will not create 'test.csv' and will test the best model with the provided test file, not using --test will create 'test.csv', resulting in 60/20/20, train dev and test respectively")
            print("k is used for the parameters of kneighbours and dtree(max depth)")
            print("d is used for the distance, in dtree is in_samples_leaf ")
            print("using np will not process the dataframe nor balance it")
            exit()
        elif opt in ('--opt'):
            ListOPT=True
            options = arg
        elif opt in ('--test'):
            hayTrain=True
            trainFile=arg;

    K=K.split(",")
    P=P.split(",")

    if path == './':
        iFile=path+str(f)
    else:
        iFile = path+"/" + str(f)
    if (path == './' and hayTrain):
        itrain=path+str(trainFile)
    elif(hayTrain):
        itrain = path+"/" + str(trainFile)
    # astype('unicode') does not work as expected
    if (ListOPT):
        listaCnfig =options.split(",")
        # print(listaCnfig)
    else:
        listaCnfig = [1, 1, 1]
    def coerce_to_unicode(x):
        if sys.version_info < (3, 0):
            if isinstance(x, str):
                return unicode(x, 'utf-8')
            else:
                return unicode(x)
        else:
            return str(x)
    if(not nproc):
        print('se va a preprocesar')
    else:
        print('no se va a preprocesar')

    #Abrir el fichero .csv y cargarlo en un dataframe de pandas
    ml_dataset = pd.read_csv(iFile)

    #comprobar que los datos se han cargado bien. Cuidado con las cabeceras, la primera línea por defecto la considerara como la que almacena los nombres de los atributos
    # comprobar los parametros por defecto del pd.read_csv en lo referente a las cabeceras si no se quiere lo comentado

    #print(ml_dataset.head(5))
    #Este apartado escribe el archivo de configuración
    df=pd.read_csv(iFile)
    listaDatos=df.columns.values
    localPath= Path(iFile)
    #CREA EL ARCHIVO DE OPCIONES
    if(ListOPT):
        config = open("config.csv", "w")
        writer = csv.writer(config)
        writer.writerow(["Feature","Use","Missing method","Scale"])
        for feature in listaDatos:
            writer.writerow([feature,int(listaCnfig[0]),int(listaCnfig[1]),int(listaCnfig[2])])
        config.close()
        if(s):
            print("se genero el archivo de configuración, saliendo")
            exit()
    try:
        config = open("config.csv","r")
    except:
        print("No hay archivo de configuración | There is no config file, please add -opt (a,b,c) when executing")
        exit()
    reader=csv.reader(config)
    ListaAEliminar=[]
    primero=True
    #AQUI LEE LAS OPCIONES, SI EL PRIMER DIGITO ES 0 DE LA FEATURE LO AÑADE A LA LISTA PARA ELIMINAR
    for feature in reader:
        if(primero):
            primero=False
        elif(int(feature[1])==0):
            print("AQUI SE ELIMINA")
            #print(ListaAEliminar)
            ListaAEliminar.append(feature[0])
    config.close()

    indice = 0
    i1 = 0
    if(not nproc):
        for a in ListaAEliminar:
            i2 = 0
            for b in listaDatos:
                if (a == b):
                    listaDatos = np.delete(listaDatos, i2, 0)
                    print("SE VA A ELIMINAR LA POS " + str(i2))
                # print(a)
                i2 = i2 + 1
            i1 = i1 + 1
    #print(listaDatos)
    ml_dataset = pd.read_csv(f,usecols=listaDatos)

    #print(ml_dataset_n)
    #print(ml_dataset)
    # Se se para la información en numericos y no numericos(categoricos y texto). Por ejemplo si es el tipo de un dato entraria dentro de categorical
    #si en cambio, es un valor como altura o, en general, valores numericos entraría en numerical data.
    #SI NO ES NUMERICO LO TRATARÁ COMO CATEGORICO Y TRANSFORMARÁ A NUMERICO
    cant_cat=0.0
    target_map={}
    #AQUI SE COMPRUEBA SI HAY TEXTO, SI HAY LO TOMARÁ COMO CATEGÓRICO Y CREARÁ
    for feature in ml_dataset:
        temporal=pd.to_numeric(ml_dataset[feature],errors='coerce')
        if(temporal.isnull().values.any()):
            print("Ha encontrado categorico")
            ml_dataset[feature]=ml_dataset[feature].astype('category')
            #print(ml_dataset[feature].dtype)
            ml_dataset[feature]=ml_dataset[feature].cat.codes
            cant_datos=ml_dataset[feature].unique()
            for x in cant_datos:#Se hace el target map, usado en la fase de training
                target_map[cant_cat] = cant_cat
                cant_cat = cant_cat + 1
        else:
            ml_dataset[feature]=temporal
    #si no se ha creado target map de categoricos se crea uno que haga categoricos de los floats ya convertidos

    #print(target_map)
    numerical_features = ml_dataset.columns
    #print(numerical_features)
    text_features = []
    #print(ml_dataset)
    categorical_features=[]
    #Estandarizamos los datos, los no numericos se les fuerza al unicode, los numericos float o fecha estandar.
    if(not nproc):
        for feature in categorical_features:
            ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

        for feature in text_features:
            ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

        for feature in numerical_features:

            if ml_dataset[feature].dtype == np.dtype('M8[ns]') or (
                    hasattr(ml_dataset[feature].dtype, 'base') and ml_dataset[feature].dtype.base == np.dtype('M8[ns]')):
                ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature])
            else:
                ml_dataset[feature] = ml_dataset[feature].astype('double')
    #Especifica la columna target, el target map es el diccionario además del tipo de dato en map.
    #target_map = {0: 0.0, 1: 1.0,3:2.0}
    ml_dataset['__target__'] = ml_dataset[target]#.map(str).map(target_map)
    #print(ml_dataset['__target__'])
    datos_dif = ml_dataset[target].unique()
    if (len(target_map) == 0):
        print(datos_dif)
        for x in datos_dif:
            target_map[x] = x
    del ml_dataset[target]
    # Remove rows for which the target is unknown.
    if(nproc):
        ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]

    #Parte los datos en dos, por un lado el test size (20%) y train (80%) si hay un train especificado en la ejecucion
    train, test = train_test_split(ml_dataset,test_size=0.2,random_state=42,stratify=ml_dataset[['__target__']])
    if(hayTrain):
        print(trainFile)
        test2=pd.read_csv(trainFile)
    else:#si no hay train se creará uno del train, de este modo tendremos 60,20,20 y se testeará al final dando un % de aciertos
        print("se crea un archivo de test")
        train, test2= train_test_split(train,test_size=0.25,random_state=42,stratify=train[['__target__']])
        test2.to_csv(index=False,path_or_buf='test.csv')
    print(train.head(5))
    #print(train['__target__'].value_counts())
    #print(test['__target__'].value_counts())
    #Si hay missing values los reemplaza, en este caso por la media.
    drop_rows_when_missing = []
    primero=True
    impute_when_missing=[]
    config = open("config.csv", "r")
    reader = csv.reader(config)
    #Aqui se carga la configuracion, modificable en el csv de configuración
    if(not nproc):#nproc es no procesar
        for feature in reader:
            if(primero or feature[0]==target or int(feature[1])==0):
                primero=False
            elif(int(feature[2])==1):
                impute="MEAN"
                entrada = {'feature': feature[0], 'impute_with': impute}
                impute_when_missing.append(entrada)
            elif(int(feature[2])==2):
                impute="MEDIAN"
                entrada = {'feature': feature[0], 'impute_with': impute}
                impute_when_missing.append(entrada)
            elif(int(feature[2])==3):
                impute="MODE"
                entrada = {'feature': feature[0], 'impute_with': impute}
                impute_when_missing.append(entrada)
            elif(int(feature[2])==4):
                impute='CREATE_CATEGORY'
                entrada = {'feature': feature[0], 'impute_with': impute}
                impute_when_missing.append(entrada)
            elif(int(feature[2])==0):
                impute="CONSTANT"
                entrada = {'feature': feature[0], 'impute_with': impute}
                impute_when_missing.append(entrada)
            else:
                print("Ha habido un error en la configuracion del archivo 'config.csv' se recomiendo borrarlo y relanzar el programa")
                exit()

    #print(impute_when_missing)
    config.close()
    # Elimina las features que haya faltantas
    if(not nproc):
        for feature in drop_rows_when_missing:
            train = train[train[feature].notnull()]
            test = test[test[feature].notnull()]
            test2=test2[test2[feature].notnull()]
            print('Dropped missing records in %s' % feature)


    if(not nproc):
     # Comprueba que opcion le hemos dado a cada feature, en el caso de opt=1 todos haran MEAN, la media.
        for feature in impute_when_missing:
            if feature['impute_with'] == 'MEAN':
                v = train[feature['feature']].mean()
            elif feature['impute_with'] == 'MEDIAN':
                v = train[feature['feature']].median()
            elif feature['impute_with'] == 'CREATE_CATEGORY':
                v = 'NULL_CATEGORY'
            elif feature['impute_with'] == 'MODE':
                v = train[feature['feature']].value_counts().index[0]
            elif feature['impute_with'] == 'CONSTANT':
                v = feature['value']
            train[feature['feature']] = train[feature['feature']].fillna(v)
            test[feature['feature']] = test[feature['feature']].fillna(v)
            test2[feature['feature']] = test2[feature['feature']].fillna(v)
            print('Imputed missing values in feature %s with value %s' % (feature['feature'], coerce_to_unicode(v)))
    rescale_features={}
    config = open("config.csv", "r")
    reader = csv.reader(config)
    primero=True
    #Carga la configuración del archivo
    if(not nproc):
        for feature in reader:
            if(primero or feature[0]==target or int(feature[1])==0):
                primero=False
            elif(int(feature[3])==1):
                rescale="AVGSTD"
                rescale_features[str(feature[0])]=(rescale)
            elif(int(feature[3])==2):
                rescale="MINMAX"
                rescale_features[str(feature[0])]=(rescale)

            else:
                print("Ha habido un error en la configuracion del archivo 'config.csv' se recomiendo borrarlo y relanzar el programa")
                exit()

    print(rescale_features)
    config.close()
    #Se especifica el reescalado,en este caso avg std
    if(not nproc):
        for (feature_name, rescale_method) in rescale_features.items():
            if rescale_method == 'MINMAX':
                _min = train[feature_name].min()
                _max = train[feature_name].max()
                scale = _max - _min
                shift = _min
            else:
                #print("entra")
                shift = train[feature_name].mean()
                scale = train[feature_name].std()
            if scale == 0.:
                del train[feature_name]
                del test[feature_name]
                del test2[feature_name]
                #print('Feature %s was dropped because it has no variance' % feature_name)
            else:
                #print('Rescaled %s' % feature_name)
                train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
                test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale
                test2[feature_name] = (test2[feature_name] - shift).astype(np.float64) / scale
    #Se separan las features y los labels.
    trainX = train.drop('__target__', axis=1)#label
    trainY = train['__target__']
    testX = test.drop('__target__', axis=1)#label
    testY = test['__target__']

    trainY = np.array(train['__target__'])#datos
    testY = np.array(test['__target__'])#datos

    if(balanced and not nproc):
    # Explica lo que se hace en este paso
        if(len(target_map)==2):
            undersample = RandomUnderSampler(sampling_strategy=0.5)# 50/50 cada clase, solo en binarios
        else:
            print("undersample auto")
            undersample = RandomUnderSampler(sampling_strategy='not minority')
        if(balanced and not(nproc)):

            trainX,trainY = undersample.fit_resample(trainX,trainY)
            testX,testY = undersample.fit_resample(testX, testY)

    resultados = open("resultados.csv", "w")
    writer = csv.writer(resultados)
    writer.writerow(["Combinación", "Precisión", "Recall", "F_score"])
    comb=1;
    f1_scoreW_max=0.0
    f1_scoreM_max=0.0
    combW=""
    combM=""
    try:
        os.mkdir("Modelos")
    except:
        print("El archivo ya existe")
    #print(trainX)
    while comb<=2:
        for k in K:
            k=int(k)
            for p in P:
                p=int(p)
                if comb <= 1:
                    mss = float(comb)
                    #print("el mss es")
                    #print(mss)
                else:
                    mss = comb
                    #print("el mss es")
                    #print(mss)
                #print("La comb es:")
                #print(k, comb, p)
                if(alg=="KNN"):

                    #Se expresan los atributos de knn, vecinos, tipo de algoritmos
                    if comb==1:

                        clf = KNeighborsClassifier(n_neighbors=k,
                                              weights='uniform',
                                              algorithm='auto',
                                              leaf_size=30,
                                              p=p)
                    else:
                        clf = KNeighborsClassifier(n_neighbors=k,
                                                   weights='distance',
                                                   algorithm='auto',
                                                   leaf_size=30,
                                                   p=p)
                    # Se especifica que estrategia se va a usar para los pesos
                elif(alg=="RFOREST"):


                    clf = RandomForestClassifier(n_estimators=100,
                                                 random_state=1337,
                                                 max_depth=k,
                                                 min_samples_leaf=p,
                                                 min_samples_split=mss,
                                                 verbose=2)
                elif(alg=="DTREE"):
                    clf = DecisionTreeClassifier(random_state=1337,
                                                 criterion='gini',
                                                 splitter='best',
                                                 max_depth=k,
                                                 min_samples_leaf=p,
                                                 min_samples_split=mss)

                if(balanced):
                    clf.class_weight = "balanced"

                # Entrenar el modelo

                clf.fit(trainX, trainY)


            # Build up our result dataset

            # The model is now trained, we can apply it to our test set:

                predictions = clf.predict(testX)
                probas = clf.predict_proba(testX)

                predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
                cols = [
                    u'probability_of_value_%s' % label
                    for (_, label) in sorted([(int(target_map[label]), label) for label in target_map])
                ]
                probabilities = pd.DataFrame(data=probas, index=testX.index, columns=cols)

            # Build scored dataset
                results_test = testX.join(predictions, how='left')
                results_test = results_test.join(probabilities, how='left')
                results_test = results_test.join(test['__target__'], how='left')
                results_test = results_test.rename(columns= {'__target__': target})

                i=0
                for real,pred in zip(testY,predictions):
                    print(real,pred)
                    i+=1
                    if i>5:
                        break
                Combinacion=("K="+str(k)+",p="+str(p))
                print(f1_score(testY, predictions, average=None))
                clasificacion=classification_report(testY,predictions,output_dict=True)
                print(classification_report(testY,predictions))
                print(confusion_matrix(testY, predictions, labels=[1,0]))


                NumTest=NumTest+1
                print("Ha finalizado el test "+str(NumTest))
                #print("el comb es"+str(comb))
                #writer.writerow([Combinacion])
                writer.writerow([""])
                writer.writerow(["Siguiente Modelo"])

                for x in clasificacion:
                    if(alg=="KNN"):
                        if(comb==1):
                            combinacion=("ALG="+alg+ "K="+str(k)+",p="+str(p)+",class="+str(x)+" Weights=Uniform")
                            combinacionArch=("ALG="+alg+ ",K="+str(k)+",p="+str(p)+ ",Weights=Uniform")
                        else:
                            combinacion = ("ALG=" + alg + "K=" + str(k) + ",p=" + str(p) + ",class=" + str(
                                x) + " Weights=Distance")
                            combinacionArch=("ALG=" + alg + ",K=" + str(k) + ",p=" + str(p) + ",Weights=Distance")
                    elif(alg=="DTREE"):
                        combinacion = ("ALG="+alg+ "Max Depth=" + str(k) + ",Min _Samples_lead=" + str(p) + ",class=" + str(
                            x) + " min_samples_split="+str(comb))
                        combinacionArch=("ALG="+alg+ ",Max Depth=" + str(k) + ",Min _Samples_lead=" + str(p) + ",min_samples_split="+str(comb))
                    elif(alg=="RFOREST"):
                        combinacion = ("ALG="+alg+ "K=" + str(k) + ",p=" + str(p) + ",class=" + str(
                            x) +" min_samples_split="+str(comb))
                        combinacionArch=("ALG="+alg+ ",K=" + str(k) + ",p=" + str(p)  +",min_samples_split="+str(comb))
                    else:
                        print("alg no identificado")
                        combinacion = ("ALG=" + alg + "K=" + str(k) + ",p=" + str(p) + ",class=" + str(
                            x) + " Weights=Uniform")
                        combinacionArch = ("ALG=" + alg + ",K=" + str(k) + ",p=" + str(p)  + ",Weights=Uniform")
                    if(x=="weighted avg"):
                        if(clasificacion[x]['f1-score']>=f1_scoreW_max):
                            f1_scoreW_max=clasificacion[x]['f1-score']
                            combW=combinacionArch
                    elif(x=="macro avg"):
                        clasificacion[x]['f1-score']
                        if(float(clasificacion[x]['f1-score']) >= f1_scoreM_max):
                            f1_scoreM_max = clasificacion[x]['f1-score']
                            combM=combinacionArch

                    try:
                        writer.writerow([combinacion,str(round(clasificacion[x]['precision'], 3)),str(round(clasificacion[x]['recall'],3)),str(round(clasificacion[x]['f1-score'],3))])
                    except:

                        if((clasificacion[x])>1):
                            writer.writerow(["Accuracy",(clasificacion[x])/100])
                        else:
                            writer.writerow((["Accuracy", round(clasificacion[x],3)]))

                nombreModel = combinacionArch+".sav"
                nombrepath = os.path.join("Modelos/", nombreModel)
                saved_model = pickle.dump(clf, open(nombrepath,"wb"))
                p = p + 1

        comb=comb+1;
    writer.writerow("")
    writer.writerow(["El mejor f1_score weighted ha sido:"])
    writer.writerow([str(f1_scoreW_max.__round__(3))+" Con la combinación:"+combW])
    writer.writerow("")
    writer.writerow(["El mejor f1_score macro ha sido:"])
    writer.writerow([str(f1_scoreM_max.__round__(3)) + " Con la combinación:" + combM])
    resultados.close()
    print("El mejor f1_score weighted ha sido:")
    print(str(f1_scoreW_max.__round__(3))+" Con la combinación:"+combW)
    print("El mejor f1_score macro ha sido:")
    print(str(f1_scoreM_max.__round__(3)) + " Con la combinación:" + combM)

    print("Probandolo con el test.csv creado automaticamente obtenemos:" )

    #print(combW+".sav")
    #print(train.columns)
    clf=pickle.load(open("Modelos/"+combW+".sav",'rb'))
    #print(test2)
    hayTarget=True
    if(not(('__target__'in test2)or(target in test2)or(trainFile=="NO"))):#CHECKEA SI HAY TARGET, SI NO HAY UNICAMENTE GUARDARA EL MODELO
        print( "Se guardará unicamente el modelo")
        hayTarget=False
    else:
        print("Weighted avg:")
        try:
            results=test2['__target__']
        except:
            test2=test2.rename(columns={target: '__target__'})
            results=test2['__target__']
        del test2['__target__']
    resultadoWeighted=clf.predict(test2)
    pd.DataFrame(resultadoWeighted).to_csv(index=False,path_or_buf='ResultadosMejorWeighted')
    if(hayTarget):
        comprobar(results,resultadoWeighted)
        print('Macro avg:')
    clf = pickle.load(open("Modelos/"+combM + ".sav", 'rb'))
    resultadoMacro=clf.predict(test2)
    if(hayTarget):
        comprobar(results,resultadoMacro)
    pd.DataFrame(resultadoWeighted).to_csv(index=False,path_or_buf='ResultadosMejorMacro')
    exit()
def comprobar(results,resultado):
    i = 0
    cont = 0
    for x in results:
        #print(x)
        #print(resultado[i])
        if (x == resultado[i]):
            cont = cont + 1
        i=i+1
    print("Aciertos:" + str(float(cont / i).__round__(4)*100)+"%")

if __name__ == '__main__':
    #print(sys.argv[1])
    KNN()
    exit()

