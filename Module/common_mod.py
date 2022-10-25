import pickle
import pandas as pd
import os           
import numpy as np
import random
import string
import re
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import itertools
import plotly.express as px
# this module contains same functions to be used in CML/ReCiPe, except one of the raw LCIA source changed from RIVM to CML

def unpickle_data (saved_data):
    unpickle_DF = None
    with open(saved_data, 'rb') as f:
        try:
            unpickle_DF = pickle.load(f)
        except pickle.UnpicklingError:
            print('Cannot write into object')
    return (unpickle_DF)


def add_rand_str_toCAS (DF_col, to_print = "no"):
    new_caslist = []
    for x in DF_col: 
        if pd.isna(x): #if x is np.nan: 
            n = random.randint(9,18)  # to be more "random", add a random int. as the string len
            xx = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(n))
            if to_print == "yes": 
                print("a new random string applied to the NaN cas_number:", xx)
            new_caslist.append(xx)
        else:
            new_caslist.append(x)
    return (new_caslist)


def uniq_catg(df, main_cat_col = "category", subcat_col = "subcategory"):
    comb_catg = list(tuple(zip(df[main_cat_col], df[subcat_col] ))) 
    uniq_catg = list(set(comb_catg))
    uniq_catg_len = len(uniq_catg)
    uniq_catg_list = list(uniq_catg)
    return ((uniq_catg_len, uniq_catg_list))  #tuple type 



#source = "openLCA", "SP", "BW2", { "RIVM" for ReCiPe, or "CML" for CML }
def change_source_catg (df, source="openLCA",  cat = "category", sub_cat = "subcategory"):
    air = df[df[cat].str.contains("air", flags=re.IGNORECASE) ] 
    common_cat_air = []
    if source == "SP" or source == "CML" or source == "RIVM":
        for value in air[sub_cat]:
            if 'low. pop.' in value:    #  SP data
                common_cat_air.append("emission/air/rural")
            elif 'rural' in value:      # RIVM data                             
                    common_cat_air.append("emission/air/rural")
            elif ("urban" in value and 'non-urban' not in value): 
                    common_cat_air.append("emission/air/urban")
            elif ("unspecified" in value):   # CML only with "unspecified" for air compartment
                common_cat_air.append("emission/air/unspecified") 
            else:
                common_cat_air.append("emission/air/others")
    elif source == "BW2" or source == "openLCA":
        for value in air[sub_cat]:
            if value == 'low population density' or value == 'non-urban air or from high stacks': 
                common_cat_air.append("emission/air/rural")
            elif value == 'low population density, long-term':
                common_cat_air.append("emission/air/rural-LT")
            elif value == 'indoor' or value == 'high population density' or value == 'urban air close to ground':
                common_cat_air.append("emission/air/urban")
            elif value ==  "unspecified":
                common_cat_air.append("emission/air/unspecified")
            else:   #e.g., openLCA has "lower stratosphere + upper troposphere" not used in others
                common_cat_air.append("emission/air/others")  
    air_newdf = air.copy()
    air_newdf["common_category"] = common_cat_air

    # for soil and water, similar source cat for openLCA and SP, thus no need if statement to seperate source DF
    soil = df[df[cat].str.contains("soil", flags=re.IGNORECASE) ]
    common_cat_soil = []
    for value in soil[sub_cat]:
        if 'agricultural' in value:
            common_cat_soil.append("emission/soil/agricultural")
        elif "unspecified" in value:
            common_cat_soil.append("emission/soil/unspecified")
        elif 'industrial' in value:
            common_cat_soil.append("emission/soil/industrial")
        elif value ==  "forestry":
            common_cat_soil.append("emission/soil/forestry")
        else: # openLCA has "urban, non industrial" for freshwater ecotoxicity
            common_cat_soil.append("emission/soil/others")
    soil_newdf = soil.copy()
    soil_newdf["common_category"] = common_cat_soil

    water = df[df[cat].str.contains("water", flags=re.IGNORECASE) ]
    common_cat_water = []
    for value in water[sub_cat]:
        if ('ocean' in value) or ('sea water' in value):   # "sea water" used for raw RIVM & raw CML
            common_cat_water.append('emission/water/ocean')
        elif ("unspecified" in value):
            common_cat_water.append('emission/water/unspecified')
        # for raw openLCA incl. a lot freshwater e.g., river/lake  
        elif ("long-term" not in value) and (("fresh" in value) or ("ground" in value) or ("surface" in value) 
                                             or ("river" in value) or ("lake" in value) or ("fossil" in value)): 
            common_cat_water.append('emission/water/freshwater')
        else: 
            common_cat_water.append("emission/water/others")  #e.g., L-T in openLCA
    water_newdf = water.copy()
    water_newdf["common_category"] = common_cat_water
    
    # emission/xxx/others will be deleted for further use: 
    air_newdf = air_newdf[air_newdf["common_category"] != "emission/air/others"]
    soil_newdf = soil_newdf[soil_newdf["common_category"] != "emission/soil/others"]
    water_newdf = water_newdf[water_newdf["common_category"] != "emission/water/others"]   
    newdf = pd.concat([air_newdf,soil_newdf,water_newdf])
    return(newdf)


def common_subcatg_tocomp (df1, df2, subcat_col = "common_category"):
    df1_uniq_catg = list(set(df1[subcat_col]))
    df2_uniq_catg = list(set(df2[subcat_col])) 
    common_uniq_catg = list(set(df1_uniq_catg).intersection(df2_uniq_catg) )
    print(len(common_uniq_catg), "unique common categories are extracted:", common_uniq_catg)
    #only extract subset of the source DF with common emission subcategory for further cor. matrix
    n = len(common_uniq_catg)   
    # create a final nested list to store all dataframe
    #[ [ [df1_commonCatg_1],[df2_commonCatg_1] ]...[ [df1_commonCatg_n],[df2_commonCatg_n] ] ]
    final_df1, final_df2, final_nested_df =  n*[1*[]],  n*[1*[]], n*[1*[]]
    for i in range(n):
        final_df1[i] = df1[df1[subcat_col] == common_uniq_catg[i]]
        final_df2[i] = df2[df2[subcat_col] ==  common_uniq_catg[i]] 
        final_nested_df[i] = [final_df1[i], final_df2[i]]
    return(final_nested_df)
    #when extracting results, final_nested_df[i][0]->common_subcatg of df1; final_nested_df[i][1]->common_subcatg of df2
    

####################################### check CFs for a same flow under same common_category ############################### 
# to check CFs for a single LCIA source, it is common for a same EF with diff. CFs under different emission compartment, but within one compartment, the CF should be the same for a same EF under the same emission compartment (e.g., common_category)
# input df is NOT the raw LCIA DF, it is the returned DF of change_source_catg() function, with a new "common_category" col
def check_EF_value_oneLCIA (df, df_EF_cat_col = "common_category", df_EF_name_col = "flow", df_EF_value_col = "value"):  
    output_DF = pd.DataFrame(columns = df.columns) 
    for x in df[df_EF_cat_col].unique():
        subset_f = df[df[df_EF_cat_col] == x]
        for f in subset_f[df_EF_name_col].unique():
            pd_list = list(subset_f[subset_f[df_EF_name_col] == f][df_EF_value_col] ) #extract value for the flow
            if (all(element == pd_list [0] for element in pd_list ) ):  # same value for the same CAS/flow within a same cat
                pass
                output_DF = output_DF.append([])
            else:
                output_DF = output_DF.append(subset_f[subset_f[df_EF_name_col] == f], ignore_index=False)  # keep orig. index  
                print("Checking by", df_EF_name_col, f, "implies different CFs under same emission compartment", x)
        if output_DF.empty:
            print("Checking by", df_EF_name_col, "same EF does not have different CFs under:", x, ", ok to be used.")
    return(output_DF)
################################# end of check CFs for a same flow under same common_category ############################### 

    
    
def compare_included_EF (df1, df1_flow_col, df2, df2_flow_col, to_print = "no"): 
    df1_uniq_flow, df2_uniq_flow = set(df1[df1_flow_col].values), set(df2[df2_flow_col].values)  # input df1_flow_col: can be "FLOW" or "CAS" 
    df1_uniq_flow_len, df2_uniq_flow_len  = len(df1_uniq_flow), len(df2_uniq_flow)
    numof_flow_in1_notin2 = len( [f for f in df1_uniq_flow  if f not in df2_uniq_flow ] )
    numof_flow_in2_notin1 = len( [f for f in df2_uniq_flow  if f not in df1_uniq_flow ] )
    if (to_print == "yes"): 
        # will print out diff. flows: 
        print( "EF in A but not in B:", [f for f in df1_uniq_flow  if f not in df2_uniq_flow ] )
        print( "EF in B but not in A:", [f for f in df2_uniq_flow  if f not in df1_uniq_flow ] )
    else:
        pass 
        #print("Detailed differences won't be printed out, to print, add input argument: to_print = 'yes'.  ")
    
    # step 0: to see detailed list of diff. between method A and B
    #return_i = [(df1_uniq_flow_len, numof_flow_in1_notin2), (df2_uniq_flow_len, numof_flow_in2_notin1)]
    # Step i: return a combined unique flow list to be used in constructing corr. matrix: 
    set_uniq_EF =  set(df1_uniq_flow).intersection(df2_uniq_flow)   #df1_uniq_flow.union(df2_uniq_flow)
    # step ii: return a summary dataframe to see diff. between method_A/_B:
    d = {"Method_A": (len(df1), df1_uniq_flow_len, numof_flow_in1_notin2), 
         "Method_B": (len(df2), df2_uniq_flow_len, numof_flow_in2_notin1), "EF_in_both": (np.nan, len(set_uniq_EF), 0)}
    summary_df = pd.DataFrame(d)
    # summary_df first index value to include the ["common_category"] name to clarify
    summary_df.index = ( (df1["common_category"].values[0] + ", N_orig_len") , "N_unique_flows", "N_flows_not_in_another")
    #print("Total intersection EF to be used is:", len(set_uniq_EF))
    return(summary_df, list(set_uniq_EF)) 
    #return([(df1_uniq_flow_len, numof_flow_in1_notin2), (df2_uniq_flow_len, numof_flow_in2_notin1)])
    
    
def final_2concat_DF (df1, df1_source, df2, df2_source, 
                      df1_flow_col="cas_number", df1_cat_col="category", df1_subcat_col="subcategory", 
                      df2_flow_col="cas_number", df2_cat_col="category", df2_subcat_col="subcategory", 
                      common_subcat_col="common_category"):
    # new_df1, new_df2 will be concat to final df1_concat/df2_concat, all EFs are sorted and CFs value can be compared directly 
    new_df1, new_df2 = [], []
    # summary_EF_table is list   summary_EF_table_i -> first returned value of function compare_included_EF()
    summary_EF_table = []
    # begin to run functions in sequence:  
    xx = change_source_catg(df1, source = df1_source, cat = df1_cat_col, sub_cat = df1_subcat_col)
    yy = change_source_catg(df2, source = df2_source, cat = df2_cat_col, sub_cat = df2_subcat_col) 
    ttt = common_subcatg_tocomp(xx, yy, subcat_col = common_subcat_col) 
    for i in range(len(ttt)): 
        summary_EF_table_i = compare_included_EF(ttt[i][0], df1_flow_col,  ttt[i][1], df2_flow_col, "no")[0]
        EF_list = compare_included_EF(ttt[i][0], df1_flow_col,  ttt[i][1], df2_flow_col, "no")[1]
        index0, index1 = [], []
        for ef in EF_list: 
        #only extract one unique index (e.g., in openLCA freshwater, river/lake/ground has multiple entries with same CF,only extract one, providing that all CFs are the same for one same "flow"/"cas_number" as checked by check_EF_value_oneLCIA() )
            index0.append (ttt[i][0].index[ttt[i][0][df1_flow_col] == ef].tolist()[0] )
            index1.append (ttt[i][1].index[ttt[i][1][df2_flow_col] == ef].tolist()[0] ) 
        new_df1_i, new_df2_i = ttt[i][0].loc[index0, : ],  ttt[i][1].loc[index1, : ]
        new_df1.append(new_df1_i)
        new_df2.append(new_df2_i)
        # summary_EF_table is the third returned value
        summary_EF_table.append(summary_EF_table_i)
        
    df1_concat = pd.concat(new_df1)
    df2_concat = pd.concat(new_df2) 
    # check if returned two final DF with same flow/category in order 
    if list(df1_concat[df1_flow_col] ) == list(df2_concat[df2_flow_col]):
        if list(df1_concat[common_subcat_col] ) == list(df2_concat[common_subcat_col] ):
            pass
    else:
        print("WARNING: returned two dataframe cannot be used for CFs comparison")
    # first two returned value are final DF for pair-wise comparison of CFs, third returned value is list of sum_table 
    return(df1_concat, df2_concat, summary_EF_table) 

    

################################### save diffEFs, w/o or w/h rounding (default 2 decimal) ######################################
### W/O rounding, most EFs ended up with diff. CF value because raw sources use diff. precision, but NO rounding error occur ###
def final_pairwise_save_diff_EF (final_df1_concat, final_df2_concat, CFvalue_col = "value", rounding = "yes", rounding_decimal = 2): 
    # change CF "value" col to float, as openLCA raw data contains x,000,000 format
    final_df1_concat[CFvalue_col] = [float(v.replace(',','')) if type(v) is str else v for v in final_df1_concat[CFvalue_col].values]
    final_df2_concat[CFvalue_col] = [float(v.replace(',','')) if type(v) is str else v for v in final_df2_concat[CFvalue_col].values]
    
    # if not rounding, then most of EFs will result in diff. CFs, but it keeps the original value and NO rounding error occur:
    if rounding == "no": 
        final_df1_concat[CFvalue_col], final_df2_concat[CFvalue_col]=final_df1_concat[CFvalue_col], final_df2_concat[CFvalue_col] 
    # combine the CF values from final_df2_concat to final_df2_concat as a new col "CFvalue_Method2"
        ndf_final = final_df1_concat.copy()
        ndf_final["CFvalue_Method2"] = final_df2_concat["value"].values
        DF_wh_diff_EF = ndf_final.loc[ndf_final[CFvalue_col] != ndf_final['CFvalue_Method2']] 
    elif rounding == "yes":     
    # if rounding: only when CFs with > two decimal points diff: 
        final_df1_concat[CFvalue_col], final_df2_concat[CFvalue_col] = round(final_df1_concat[CFvalue_col], rounding_decimal), round(final_df2_concat[CFvalue_col], rounding_decimal)
    # combine the CF values from final_df2_concat to final_df2_concat as a new col "CFvalue_Method2"
        ndf_final = final_df1_concat.copy()
        ndf_final["CFvalue_Method2"] = final_df2_concat["value"].values
        DF_wh_diff_EF = ndf_final.loc[ndf_final[CFvalue_col] != ndf_final['CFvalue_Method2']] 
    else: 
        print ("Define input parameter <rounding> either as <yes>yes or <no>, default input parameter: rounding = 'yes' and rounding_decimal = 2. ")
    return (DF_wh_diff_EF)

########################################### END of save diffEFs, w/o or w/h rounding ########################################
    


####################################################### corrlation cal.  #######################################################
# pearson corr.  assumptions: constant variance and linearity
def final_pairwise_CF_corr (final_df1_concat, final_df2_concat, CFvalue_col = "value" ): 
    fv1 = [float(v.replace(',','')) if type(v) is str else v for v in final_df1_concat[CFvalue_col].values]
    fv2 = [float(v.replace(',','')) if type(v) is str else v for v in final_df2_concat[CFvalue_col].values] 
    correlation, p_value = pearsonr(fv1, fv2)
    return (correlation)

# spearman corr.
def final_pairwise_CF_corr_spearman (final_df1_concat, final_df2_concat, CFvalue_col = "value" ): 
    fv1 = [float(v.replace(',','')) if type(v) is str else v for v in final_df1_concat[CFvalue_col].values]
    fv2 = [float(v.replace(',','')) if type(v) is str else v for v in final_df2_concat[CFvalue_col].values] 
    correlation, p_value = spearmanr(fv1, fv2)
    return (correlation)
################################################## END of corrlation cal. ######################################################




# to plot out all EFs with diff. CFs and save as HTML (input: df1/2_name: used in plotting, DF_wh_diff_EF is resulting diff_EF)
def final_diff_EF_plot2 (lcia_name, df1_name, df2_name, DF_wh_diff_EF, flow_col = "cas_number", value1_col = "value", value2_col = "CFvalue_Method2" ):
    # DF_wh_diff_EF is result from the function final_pairwise_save_diff_EF()
    diff_EFs_plot = DF_wh_diff_EF[[flow_col, "common_category",  value1_col, value2_col]].copy()
    diff_num = diff_EFs_plot[value1_col].values / diff_EFs_plot[value2_col].values
    #diff_num2 = [round(x,2) for x in diff_num]
    new_col_plot = str(df1_name + "_CF / " + df2_name + "_CF")
    diff_EFs_plot[new_col_plot] = diff_num
    #diff_EFs_plot = diff_EFs_plot.sort_values([new_col_plot], ascending = ( False)) 
    fig = px.bar(diff_EFs_plot, x=flow_col, y = new_col_plot, color="common_category",    #orientation='h'
                 title= str(lcia_name + "_EFs w/h diff CFs: " + df1_name + "÷" + df2_name)).update_xaxes(categoryorder="total descending")
    fig.show()
    # to save HTML figures: 
    #if not os.path.exists("results/Emissions_IC/diff_EFs_HTMLimage"):
    #    os.makedirs("results/Emissions_IC/diff_EFs_HTMLimage")
    #filename = lcia_name + "_" + df1_name + "_divided_by_" + df2_name + ".html"
    #fig.write_html("results/Emissions_IC/diff_EFs_HTMLimage/" + filename)
    return (fig)

