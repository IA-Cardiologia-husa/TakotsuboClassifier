# Takotsubo vs Control Classifier

This code provides the training and evaluation of the models for the research paper *The Role of Spectral CT in Stress Cardiomyopathy: Pathophysiological Insights and Comprehensive Diagnosis*. To execute the code one needs to create a Python environment with the following libraries installed (version used in parentheses):
- python (3.11.5)
- numpy (1.25.2)
- pandas (2.1.1)
- matplotlib (3.8.0)
- scipy (1.11.3)
- openpyxl (3.1.2)
- xlsxwriter (3.1.6)
- luigi (3.4.0)
- scikit-learn (1.3.1)
- xgboost (2.0.0)
- shap (0.42.1)

The code can be run with the following command:

	python -m luigi --module KoopaML AllTasks --all-shap Yes --local-scheduler
	
This will create several folders. ML algorithms will be placed as .pickle files in *models* folder. Validation results can be seen in the *report* folder. The original data to train the models is not provided, but a *fakedata.csv* file is used instead to simulate the training. The actual models described in the research article, and the evaluation report, can be found in the folders *final models* and *final report* respectively.

Data ingestion can be modified in the file *user_data_utils.py*, ML algorithms can be specified in the file *user_MLmodels_info.py*, and the workflow of training/validation can be specified in *user_Workflow_info.py*.

The variables used in the model are the following ones:

- '1P_Total_Mediana_Yodo_80%': Median of the iodine density of the myocardium in the first pass perfusion CT acquisition, discarding the inner and outer 10% of the voxels.
- '1P_Total_Volumen': Volume of the myocardium as measured in the first pass perfusion CT acquisition
- '1P_Gradiente_Apical_Basal': Apex-to-base iodine density gradient in the first pass perfusion CT acquisition (i.e. median of iodine density in the apical myocardium segments minus median of the iodine density in the basal myocardium segments)
- '1P_Gradiente_Medio_Basal': Mid-cavity-to-base iodine density gradient in the first pass perfusion CT acquisition
- '1P_Aortic_Blood_Pool': Median of the iodine density of the blood pool of the aorta in the first pass perfusion CT acquisition (For this, we considered only aorta voxels within 20% of the long axis length away from the aortic valve, and then only the inner 50% of voxels)
- 'PCAT_LAD': Average Pericoronary Adipose Tissue Attenuation in the Left Anterior Descent coronary artery as measured by Philips Healthcare *PCAT research tool*
- 'PCAT_LCx': Average Pericoronary Adipose Tissue Attenuation in the Left Circumflex coronary artery as measured by Philips Healthcare *PCAT research tool*
- 'PCAT_RCA': Average Pericoronary Adipose Tissue Attenuation in the Right coronary artery as measured by Philips Healthcare *PCAT research tool*
- 'RT_Total_Mediana_Yodo_80%': Median of the iodine density of the myocardium in the late enhancement CT acquisition, discarding the inner and outer 10% of the voxels.
- 'RT_Total_Volumen': Volume of the myocardium as measured in the late enhancement CT acquisition
- 'RT_Gradiente_Apical_Basal'': Apex-to-base iodine density gradient in the late enhancement CT acquisition
- 'RT_Gradiente_Medio_Basal': Mid-cavity-to-base iodine density gradient in the late enhancement CT acquisition
- 'RT_Aortic_Blood_Pool': Median of the iodine density of the blood pool of the aorta in the late enhancement CT acquisition
- 'PCAT_Total': Average Pericoronary Adipose Tissue Attenuation of the whole coronary tree
- 'ECV_CT_Total': Median of the Extracellular volume of the myocardium, measured in the late enhancement CT acquisition
