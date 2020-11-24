def cluster_tercil(df,coluna):
    #df = df[df['Nome_da_coluna']==1]
    
    prim_tercil = df[df['Nome_da_coluna']==1][coluna].quantile(q=0.33)
    seg_tercil = df[df['Nome_da_coluna']==1][coluna].quantile(q=0.66)
    
    df[coluna+'_cluster'] = np.select([((df[coluna] <=  prim_tercil)),
                                           ((df[coluna] > prim_tercil)&(df[coluna] <= seg_tercil)),
                                             (df[coluna] > seg_tercil)],
                                             
                                          ['baixo','medio','alto'],
                                          'sem_informacao')

    return df[coluna+'_cluster']
    
    
def regressao_logistica(df, string_model = ''):
   
    
    #aplica regressão geral nos df_1
    est = smf.logit(formula = string_model, data=df)
    lr = est.fit()
   
    #cria dataframe com os coeficientes
    coeficientes = pd.DataFrame(lr.params).T
    coeficientes['indicador'] = 'coeficientes'
   
    #cria dataframe com os p-valores
    p_values = pd.DataFrame(lr.pvalues).T
    p_values['indicador'] = 'p-value'
   
    #faz append dos coeficientes e dos p-valores para observamos todos juntos num dataframe
    coeficientes = coeficientes.append(p_values)
    
    #Cria objeto com efeitos marginais da logistica
    efeitos_marginais = lr.get_margeff(at='overall', method='dydx', atexog=None, dummy=False, count=False).summary()
    return(print(lr.summary()), print(efeitos_marginais))
    
    
    
def regressao_linear(df, string_model = ''):
   
    #aplica regressão geral nos df_1
    est = smf.ols(formula = string_model, data=df)
    lr = est.fit()
   
    #cria dataframe com os coeficientes
    coeficientes = pd.DataFrame(lr.params).T
    coeficientes['indicador'] = 'coeficientes'
   
    #cria dataframe com os p-valores
    p_values = pd.DataFrame(lr.pvalues).T
    p_values['indicador'] = 'p-value'
   
    #faz append dos coeficientes e dos p-valores para observamos todos juntos num dataframe
    coeficientes = coeficientes.append(p_values)
  
    return(print(lr.summary()))
