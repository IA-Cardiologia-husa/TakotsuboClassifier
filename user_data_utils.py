#Import libraries
import pandas as pd
import numpy as np


def load_database():
	df = pd.read_csv("fakedata.csv")
	return df

def clean_database(df):
	df = df.T.drop_duplicates().T
	return df

def process_database(df, wf_name):
	for v in df.columns[:-1]:
		try:
			df[v] = df[v].astype(float)
		except:
			df.loc[~df[v].apply(type).isin([int,float]),v] = np.nan
			df[v] = df[v].astype(float)
	return df

def load_external_database():
	# df = pd.read_excel("External Database.xlsx")
	return df

def clean_external_database(df):
	df = clean_database(df)
	return df

def process_external_database(df, wf_name):
	df = process_database(df, wf_name)
	return df
