# In this archive we have to define the dictionary WF_info. This is a dictionary of dictionaries, that for each of our workflows
# assigns a dictionary that contains:
#
# formal_title:		Title for plots and reports
# label_name:		Variable to predict in this workflow, e.g.: 'Var17'
# feature_list:		List of features to use in the ML models, e.g.: ['Var1', 'Var2', 'Var4']
# filter_function:	Function to filter the Dataframe. If we want to use only the subject of the dataframe with Var3=1,
# 					we would write: lambda df: df.loc[df['Var3']==1].reset_index(drop=True)
# 					In case we want no filter, we have to write: lambda df: df
#  					This is an alterative to WF-specific code in process_database()
# group_label:		groups for cross-validation. Subjects from the same groups
#					will appear in the same folds
# validation_type: 	"kfold", "groupkfold" (repetitions don't work in this scheme), "stratifiedkfold", "stratifiedgroupkfold",
#					"unfilterdkfold" (for doing the kfold first and then filtering the folds)
# cv_folds:			For kfolds, the number of folds
# cv_repetitions:	For kfolds, the number of repetitions
# external_validation: 'Yes' or 'No', in case of 'Yes', you have to fill user_external_data_utils.py
#
# Example:
#

WF_info ={}

WF_info['FullModel'] = {'formal_title': 'Tako-Tsubo vs Control (All CT Variables)',
						  'type': 'classification',
						  'label_name': 'Tako',
						  'label_time': None,
						  'feature_list': ['1P_Total_Mediana_Yodo_80%',
											 '1P_Total_Volumen',
											 '1P_Gradiente_Apical_Basal',
											 '1P_Gradiente_Medio_Basal',
											 '1P_Aortic_Blood_Pool',
											 'PCAT_LAD',
											 'PCAT_LCx',
											 'PCAT_RCA',
											 'RT_Total_Mediana_Yodo_80%',
											 'RT_Total_Volumen',
											 'RT_Gradiente_Apical_Basal',
											 'RT_Gradiente_Medio_Basal',
											 'RT_Aortic_Blood_Pool',
											 'PCAT_Total',
											 'ECV_CT_Total'],
						  'filter_function': lambda df: df,
						  'models':['BT', 'RF', 'LR_SCL_HypTuning'],
						  'group_label': None,
						  'validation_type':'kfold',
						  'cv_folds': 3,
						  'cv_repetitions': 10,
						  'external_validation': 'No',
						  'filter_external_validation': lambda df: df,
						  'metrics':['aucroc','aucroc_pooled','aucpr']}

WF_info['SimpleModel'] = {'formal_title': 'Tako-Tsubo vs Control (4 variables)',
						  'type': 'classification',
						  'label_name': 'Tako',
						  'label_time': None,
						  'feature_list': ['PCAT_Total',
											'ECV_CT_Total',
											'1P_Total_Volumen',
											'1P_Gradiente_Apical_Basal'],
						  'filter_function': lambda df: df,
						  'models':['BT', 'RF', 'LR_SCL_HypTuning'],
						  'group_label': None,
						  'validation_type':'kfold',
						  'cv_folds': 3,
						  'cv_repetitions': 10,
						  'external_validation': 'No',
						  'filter_external_validation': lambda df: df,
						  'metrics':['aucroc','aucroc_pooled','aucpr']}
