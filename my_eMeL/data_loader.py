import pandas as pd





def load_known_txt( file_path, delimiter, data_column_asList, label_column ):
	f = open(file_path,'r')
	data = {}
	for line in f.readlines():
		parts = line.split(delimiter)
		for i,onePart in enumerate(parts):
			onePart = onePart.replace('\r','').replace('\n','')
			column_name = 'col'+str(i)
			try:
				data[column_name].append(onePart)
			except Exception as e:
				data[column_name] = []
				data[column_name].append(onePart)
	df = pd.DataFrame(data)
	data_column_asList = [ 'col'+str(columnNumber) for columnNumber in data_column_asList]
	label_column = 'col'+str(label_column)

	data_df = df[data_column_asList]
	#data_df = data_df.to_frame() 

	label_df = df[label_column]


	map_ = {}
	for i,l in enumerate(list(label_df.unique())):
		map_[l] = str((i+1)) 
	label_df = label_df.map(map_)
	
	label_df = label_df.to_frame()
	label_df.columns = ['Labels']
	

	return data_df,label_df

''' or 
# import some data
iris_df = data_loader.load_txt( file_path    = 'path',
                                delimiter    = ','    )
# take a look to data 
iris_df
# split data
iris_data_df, iris_label_df = iris_df[['col1','col4']],iris_df[['col5']]
'''