# Seleccionador de variables
Este es el proyecto de seleccionador de variables para problemas de regresión o clasificación, utiliza una tabla con variable respuesta y covariables que permite obtener un conjunto de covariables acotado al original.


## Metodologías

Este aglomerado utiliza varias funcionalidades para la seleccion de variables, entre ellas tenemos:

### Bosques aleatorios y Regresión Ridge
```python
res_sel_features = sel.mejores_variables_rf(X_base_con_imputaciones,
                                        np.ravel(y_base_con_imputaciones),
                                        2, ruidos = 1, ridge_flag = 1, folds=3, seed = 42, type = "class")
```

Este permite calcular la importancia relativa de varios bosques aleatorios *débiles*, la importancia de las covariables d euna regresión ridge y añade un par de ruidos aleatorios, devuelve un dataframe con el ranking general de las importancias de las variables.

### Boruta
```python
boruta_vars = sel.Boruta(X_base_con_imputaciones,y_base_con_imputaciones, type = "class")
```

La metodología Boruta calcula la importancia de las variables según la proporción de ocaciones que una variable sale más importante que su *sombra* en un bosque aleatorio durante varias iteraciones.

### Information Value y WOE
```python
sel.iv_woe(base_train_sample,"var_resp",bins=10, show_woe=False)
```
Esta metodología esta implementada actualmente para problemas de clasificación, pues binea el rango de las variables para ver su importancia con la clasificacion de la respuesta.

### Correlación
```python
corr_par = sel.correlaciones_parejas( nombre_iteracion, base_sin_imputaciones, variables_categoricas)
```

se calculan las correlaciones lineales de la base de datos para eventualmente depurar la base usando la metodologia de *res_sel_features*
```python
varsalen = sel.depuracion_vars_correlacionadas_corr( nombre_iteracion, corr_par, res_sel_features )
feats_train = [e for e in base_sin_imputaciones.columns if e not in  list(varsalen['variable'])]
feats_train = list(set(feats_train))
feats_train
```









