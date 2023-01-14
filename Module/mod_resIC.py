# functions used for resource impact categories (resource-ICs)
import pickle
import pandas as pd
import os           
import numpy as np
import re
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import itertools
import plotly.express as px
import random 
import string


def unpickle_data (saved_data):
    unpickle_DF = None
    with open(saved_data, 'rb') as f:
        try:
            unpickle_DF = pickle.load(f)
        except pickle.UnpicklingError:
            print('Cannot write into object')
    return (unpickle_DF)


def res_compare_included_EF (df1, df1_flow_col, df2, df2_flow_col, to_print = "no"): 
    df1_uniq_flow, df2_uniq_flow = set(df1[df1_flow_col].values), set(df2[df2_flow_col].values)  # input df1_flow_col: can be "FLOW" or "CAS" 
    df1_uniq_flow_len, df2_uniq_flow_len  = len(df1_uniq_flow), len(df2_uniq_flow)
    numof_flow_in1_notin2 = len( [f for f in df1_uniq_flow  if f not in df2_uniq_flow ] )
    numof_flow_in2_notin1 = len( [f for f in df2_uniq_flow  if f not in df1_uniq_flow ] )
    if (to_print == "yes"): 
        #to print out diff. flows: 
        print( "EF in A but not in B:", [f for f in df1_uniq_flow  if f not in df2_uniq_flow ] )
        print( "EF in B but not in A:", [f for f in df2_uniq_flow  if f not in df1_uniq_flow ] )
    else:
        print("Detailed differences won't be printed out, to print, add input argument: to_print = 'yes'.  ")
    # step i: see diff. between method A and B
    comp_EF_outputDF = pd.DataFrame(np.array([[df1_uniq_flow_len, df2_uniq_flow_len ], [numof_flow_in1_notin2, numof_flow_in2_notin1 ] ]),
                   columns=['Method_A', 'Method_B'])
    comp_EF_outputDF.index = ['total_uniq_EF', 'EF_notin_another']
    
    # Step ii: return a unique flow list to be used in constructing corr. matrix: 
    comb_uniq_EF = df1_uniq_flow.union(df2_uniq_flow)
    set_uniq_EF =  set(df1_uniq_flow).intersection(df2_uniq_flow) 
    print("N of common EFs to be used is:", len(set_uniq_EF))   
    #print("Combined (unions) EF is:", len(comb_uniq_EF))
    return(comp_EF_outputDF, set_uniq_EF)


def check_EF_value (df, df_EF_name_col, df_EF_value_col, EF_list):  #df_EF_col_name is "FLOW" to check agains EF_list which is second return value of compare_included_EF() function
    output_DF = pd.DataFrame(columns = df.columns) 
    for f in EF_list:
        subset_f = df[df[df_EF_name_col] == f]
        pd_list = list(df[df[df_EF_name_col] == f][df_EF_value_col] ) #extract value for the flow
        if (all(element == pd_list [0] for element in pd_list ) ):  # same value for the same CAS/flow regardless of (sub)compartment
            pass
            output_DF = output_DF.append([])
        else:
            print(subset_f[df_EF_name_col].values,  subset_f[df_EF_value_col].values)
            output_DF = output_DF.append(subset_f, ignore_index=True)  
    return(output_DF)



# final calculation1
def append_EF_value (df, df_EF_name_col, df_EF_value_col, EF_list):
    Method_x_value = []    
    for f in EF_list:
        if (len(df[df[df_EF_name_col] == f][df_EF_value_col] )) != 0:   # has the same flow
            pd_list = list(df[df[df_EF_name_col] == f][df_EF_value_col] )
            if (all(element == pd_list[0] for element in pd_list ) ):  # same value for the same CAS/flow regardless of (sub)compartment
                value_str = str(pd_list[0])
                Method_x_value.append(float(value_str.replace(',','') ) )
            else:
                print("Different value for a same flow:", f)
                Method_x_value.append(np.nan)
        else:                                                          # no corresponding flow
            Method_x_value.append(np.nan)
    return(Method_x_value)

# final calculation2
def final_EF_combined (df1, df1_EF_name_col, df1_EF_value_col, df2, df2_EF_name_col, df2_EF_value_col, EF_list):
    Method_A_value = append_EF_value(df1, df1_EF_name_col, df1_EF_value_col, EF_list)
    Method_B_value = append_EF_value(df2, df2_EF_name_col, df2_EF_value_col, EF_list)
    return(Method_A_value, Method_B_value)