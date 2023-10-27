import sys
import time
import os
from pathlib import Path  
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from boruta import BorutaPy
import pandas as pd
import datetime
import numpy as np
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
import seaborn as sn
import random
from random import randint
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV, RidgeCV



def mejores_variables_rf(X, y, iteraciones_rf,type = "reg", ruidos=0, ridge_flag=0, folds=10, seed=42):
    """
    Califica la importancia de variables en un dataset con variables dependientes e
    independiente definida, se utiliza para preseleccionar variables de un módulo de información.

    Dependencies:
        Impala_Helper, su diccionario cache y hp = Helper( cache )

    Args:
        tabla_entrada (str): data frame pandas que contiene los datos
        cache (dic): Diccionario de parámetros (cache) que utiliza el Impala_Helper
        folds (int): Cantidad de folds para el entrenamiento

    Returns:
        Un dataframe de variables dependientes con importancia calificada, en LZ

    """
    # Define variables independientes
    print("--- Preparando tablas para el modelamiento ... ---")
    start_time = datetime.datetime.now()

    # Crea variables de ruido normal y uniforme
    if ruidos == 1:
        X["random_normal"] = np.random.normal(0, 1, size=(X.shape[0]))
        X["random_uniforme"] = np.random.uniform(0, 1, size=(X.shape[0]))

    print("    %s runtime " % (datetime.datetime.now() - start_time))

    """ ENSAMBLE DÉBIL """
    # Define y entrena ensamble débil
    print("--- RandomForest - Seleccionando y entrenando el mejor modelo con GridSearchCV ... ---")
    start_time = datetime.datetime.now()
    multiplicador = 1
    
    np.random.seed(seed)
    for i in range(iteraciones_rf):
        print(i)
        if i == 0: 
            rn = 5
        else:
            multiplicador = multiplicador * 10
            rn = rn*5
        
        #         seleccionar aleatoriamente el numero de variables a considerar por el rf
    #        mf = randint(int(len(X.columns) * 0.3), len(X.columns))
    
        print('rn:' + str(rn))
    #        print('mf:' + str(mf))
    
        print(i)
        rf_param_grid = {
            'n_estimators': [5, 10 * multiplicador],
            'max_depth': [rn],
            'random_state': [seed]*iteraciones_rf,
            #                      'max_features': [mf],
            'n_jobs': [-1]
        }
        
        if type == "class":
            np.random.seed(seed)
            rf = RandomForestClassifier()
            rf_grid_cv = GridSearchCV(rf, rf_param_grid, cv=folds, verbose=2)
            print("    %s runtime " % (datetime.datetime.now() - start_time))
            rf_grid_cv.fit(X, y)
        else:
            np.random.seed(seed)
            rf = RandomForestRegressor()
            rf_grid_cv = GridSearchCV(rf, rf_param_grid, cv=folds, verbose=2)
            print("    %s runtime " % (datetime.datetime.now() - start_time))
            rf_grid_cv.fit(X, y)   
    
            # Atributos del mejor modelo
            rf_best = rf_grid_cv.best_estimator_
    
            # Importancia variables
            rf_imp = pd.Series(rf_best.feature_importances_, index=X.columns).sort_values(ascending=False)
            df_rf_imp = pd.DataFrame({'variable': rf_imp.index, 'importancia_rf_' + str(i): rf_imp.values})
            df_rf_imp['rank_rf_' + str(i)] = df_rf_imp['importancia_rf_' + str(i)].rank(method="max")
    
            print("    %s runtime " % (datetime.datetime.now() - start_time))
    
            if i == 0:
                df_importancias = df_rf_imp
                df_importancias['importancia'] = df_importancias['rank_rf_' + str(i)]
            else:
                df_importancias = pd.merge(df_importancias, df_rf_imp, on="variable")
                df_importancias['importancia'] = df_importancias['importancia'] + df_importancias['rank_rf_' + str(i)]
    
    np.random.seed(seed)
    if ridge_flag == 1 and type=="class":
        print("--- Ridge  ---")
        """ REGRESIÓN RIDGE """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        ridge = RidgeClassifierCV(cv=folds)
        ridge.fit(X_scaled, y)
        ridge_imp = pd.Series(ridge.coef_.flatten(), index=X.columns)
        df_ridge_imp = pd.DataFrame({'variable': ridge_imp.index, 'importancia_ridge': ridge_imp.values})
        print("    %s runtime " % (datetime.datetime.now() - start_time))
        df_importancias = pd.merge(df_importancias, df_ridge_imp, on="variable")
        df_importancias['rank_ridge'] = df_importancias.importancia_ridge.rank(method="max")
        df_importancias['importancia'] = df_importancias['importancia'] + df_importancias['rank_ridge']
    elif ridge_flag == 1 and type=="reg":
        print("--- Ridge  ---")
        """ REGRESIÓN RIDGE """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        ridge = RidgeCV(cv=folds)
        ridge.fit(X_scaled, y)
        ridge_imp = pd.Series(ridge.coef_.flatten(), index=X.columns)
        df_ridge_imp = pd.DataFrame({'variable': ridge_imp.index, 'importancia_ridge': ridge_imp.values})
        print("    %s runtime " % (datetime.datetime.now() - start_time))
        df_importancias = pd.merge(df_importancias, df_ridge_imp, on="variable")
        df_importancias['rank_ridge'] = df_importancias.importancia_ridge.rank(method="max")
        df_importancias['importancia'] = df_importancias['importancia'] + df_importancias['rank_ridge']

    df_importancias = df_importancias.sort_values(by='importancia', ascending=False)

    """ EXPORTA DATOS A CSV """
    #     now = datetime.datetime.now()
    #     df_importancias.to_csv('seleccion_vars_{}_{}.csv'.format(modulo,(now.strftime('%b%d%H%M'))), index=False)
    #     print('La preselección fue exportada a seleccion_vars_{}_{}.csv en el directorio de trabajo'.format(modulo,(now.strftime('%b%d%H%M'))))
    return df_importancias

class Boruta:
    """
    Califica la importancia de variables en un dataset con a travéz del método de boruta

    Dependencies:
        Impala_Helper, su diccionario cache y hp = Helper( cache )

    Args:
        X (DataFrame): data frame pandas que contiene los datos 
        y (Dataframe): data frame con la variable respuesta
        max_depth (int): valor de la profundidad máxima de el bosque
        max_iter (int): iteracion maxima de la corrida del algoritmo
        type (chr): "reg" o "class" si es un problema de clasificacion o regresion

    Returns:
        Un dataframe de variables dependientes con importancia calificada

    """
    def __init__(self,X,y,max_depth=5,max_iter=100, type = "reg"):
        ### iniciar boruta
        if type=="reg":
            forest = RandomForestRegressor(max_depth=max_depth)
            boruta = BorutaPy(
                estimator = forest,
                n_estimators = "auto",
                max_iter = max_iter # numero de intentos a realizar
            )
            # Ajustar boruta (recibe np.array, No DataFrames)
            boruta.fit(np.array(X),np.array(y))
            
            # Resultados
            self.vars_aceptadas = X.columns[boruta.support_].to_list()
            self.vars_debiles = X.columns[boruta.support_weak_].to_list()
            self.vars_basura = X.columns.drop(self.vars_aceptadas+self.vars_debiles).to_list()
        
        elif type=="class":
            ### iniciar boruta
            forest = RandomForestClassifier(max_depth=max_depth)
            boruta = BorutaPy(
                estimator = forest,
                n_estimators = "auto",
                max_iter = max_iter # numero de intentos a realizar
            )
            # Ajustar boruta (recibe np.array, No DataFrames)
            boruta.fit(np.array(X),np.array(y))
            
            # Resultados
            self.vars_aceptadas = X.columns[boruta.support_].to_list()
            self.vars_debiles = X.columns[boruta.support_weak_].to_list()
            self.vars_basura = X.columns.drop(self.vars_aceptadas+self.vars_debiles).to_list()


def iv_woe(data, target, bins=10, show_woe=False):
    
    # DataFrame vacio
    newDF,woeDF = pd.DataFrame(), pd.DataFrame()
    
    #Nombres de las columnas
    cols = data.columns
    
    # Correr el WOE y el IV a todas las variables del set
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})

        
        # Calcular el numero de eventos en cada grupo (bin)
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        
        # Calcular % de eventos en cada grupo.
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()

        # calcular los no eventos en cada grupo.
        d['Non-Events'] = d['N'] - d['Events']
        # Calcular el % de no eventos en cada grupo.
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()

        # Calcular el WOE tomando el log natural de la division entre % de no-eventos y % de eventos
        d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        print("IV de la variable " + ivars + " es: " + str(round(d['IV'].sum(),6)))
        temp =pd.DataFrame({"Variable" : [ivars], "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
        newDF=pd.concat([newDF,temp], axis=0)
        woeDF=pd.concat([woeDF,d], axis=0)

        #Mostrar la Tabla de WOE
        if show_woe == True:
            print(d)
    return newDF, woeDF


def correlaciones_parejas(experimento, df, variables_categoricas):
    import pandas as pd
    import datetime
    from dython.nominal import associations
    
    print('--- Calculando correlaciones ---')
    corr = associations(df, nominal_columns = variables_categoricas, plot=False)
    corr_df = corr['corr']
    
    # Reporte de resultados: se guardan sólo las parejas de variables con correlación superior a 0.7
    print('Escribiendo reporte...')
    resultados  = pd.DataFrame(columns = ['variable1','variable2','correlacion'])
    n_vars = corr_df.shape[0]
    variab = corr_df.columns
    for f in range( n_vars ):
        for c in range( f+1, n_vars ):
            cor_fc = corr_df.iloc[f, c]
            if abs( cor_fc ) >= 0.8 :
                res = pd.DataFrame.from_dict({'variable1' : [variab[f]], 'variable2' : [variab[c]], 'correlacion' : round(cor_fc,4)})
                resultados = resultados.append( res )
    resultados.to_csv('corr_parejas_' + experimento + '_{}.csv'.format(datetime.datetime.now().strftime('%b%d%H%M')))
    return resultados



# Itera para arrojar las variables a eliminar
def depuracion_vars_correlacionadas_corr(experimento, res_correlaciones, res_importancias):

    # Se hace el cruce con la tabla de importancias consolidada a partir de los resultados de los ensambles débiles
    correlaciones = res_correlaciones[['correlacion', 'variable1', 'variable2']]
    importancias = res_importancias[['importancia', 'variable']]
    # Cruza con variable 1
    importancias.columns = ['imp1', 'variable1']
    resultados_importancias = correlaciones.merge(importancias, how='left')

    # Cruza con variable 2
    importancias.columns = ['imp2', 'variable2']
    resultados_importancias = resultados_importancias.merge(importancias, how='left')

    # Determina la variable con mayor importancia
    resultados_importancias['var'] = resultados_importancias[['imp1', 'imp2']].idxmax(axis=1)

    # Se define la siguiente función para armar una lista con objetos únicos a partir de la concatenación de dos listas
    def concat_variables(grupo1, grupo2):
        x1 = pd.DataFrame(grupo1).append(pd.DataFrame(grupo2))
        return x1.iloc[:, 0].unique().tolist()

        # Paso 1 : parte a - se seleccionan en vars_t las variables que hayan ganado en importancia

    vars1 = resultados_importancias['variable1'][resultados_importancias['var'] == 'imp1'].unique()
    vars2 = resultados_importancias['variable2'][resultados_importancias['var'] == 'imp2'].unique()
    vars_t = concat_variables(vars1, vars2)

    # Paso 1 : parte b - se determina cuáles variables están dentro de vars_t
    resultados_importancias['a1'] = resultados_importancias['variable1'].apply(lambda x: x in vars_t)
    resultados_importancias['a2'] = resultados_importancias['variable2'].apply(lambda x: x in vars_t)

    # Paso 1 : parte c - se construye a3 que es True cuando las dos variables fueron seleccionadas
    resultados_importancias['a3'] = resultados_importancias['a1'] & resultados_importancias['a2']
    resultados_importancias['a3_prev'] = resultados_importancias['a3']

    # Paso 1 : parte d - se escogen las variables que salen, que son las que son False en a3, y además son la de menor importancia
    vars_salen_imp1 = resultados_importancias['variable1'][
        (np.logical_not(resultados_importancias['a3'])) & (resultados_importancias['var'] == 'imp2')].tolist()
    vars_salen_imp2 = resultados_importancias['variable2'][
        (np.logical_not(resultados_importancias['a3'])) & (resultados_importancias['var'] == 'imp1')].tolist()
    vars_salen = pd.DataFrame(concat_variables(vars_salen_imp1, vars_salen_imp2), columns=['variable'])
    vars_salen['ronda'] = 1

    # En el while se hacen los demás pasos
    cond = True
    max_iters = 1000
    contador = 1
    while (cond):

        contador = contador + 1

        # parte a
        vars1 = resultados_importancias['variable1'][
            (resultados_importancias['var'] == 'imp1') & (resultados_importancias['a3'])].unique()
        vars2 = resultados_importancias['variable2'][
            (resultados_importancias['var'] == 'imp2') & (resultados_importancias['a3'])].unique()
        vars_t = concat_variables(vars1, vars2)

        # parte b
        resultados_importancias['a1'] = resultados_importancias['variable1'].apply(lambda x: x in vars_t)
        resultados_importancias['a2'] = resultados_importancias['variable2'].apply(lambda x: x in vars_t)

        # parte c
        resultados_importancias['a3'] = resultados_importancias['a3'] & resultados_importancias['a1'] & \
                                        resultados_importancias['a2']

        # parte d
        vars_salen_imp1 = resultados_importancias['variable1'][
            (resultados_importancias['a3_prev'] & np.logical_not(resultados_importancias['a3'])) & (
                        resultados_importancias['var'] == 'imp2')].tolist()
        vars_salen_imp2 = resultados_importancias['variable2'][
            (resultados_importancias['a3_prev'] & np.logical_not(resultados_importancias['a3'])) & (
                        resultados_importancias['var'] == 'imp1')].tolist()
        vars_salen_i = pd.DataFrame(concat_variables(vars_salen_imp1, vars_salen_imp2), columns=['variable'])
        vars_salen_i['ronda'] = contador

        resultados_importancias['a3_prev'] = resultados_importancias['a3']

        vars_salen = vars_salen.append(vars_salen_i)

        print('Iteracion ' + str(contador))

        # se sale del loop cuando se alcanza el numeró máximo de iteraciones, o cuando ya no hay pares de variables correlacionadas
        if (contador == max_iters or sum(resultados_importancias['a3_prev']) == 0):
            cond = False

    print(vars_salen)
    return vars_salen
