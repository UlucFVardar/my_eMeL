import pandas as pd

class AccuracyTable():
    def __init__(self,index, columns):
        self.data = {}
        self.index = index
        self.columns = columns
    def create_table_template(self):
        template = {}
        template[self.index] = []
        for c in self.columns:
            template[c] = []
        return template    
    
    def check_sub_table(self,header_name):
        try:
            d = self.data[header_name]
        except Exception as e:
            self.data[header_name] = self.create_table_template()
    
    def add_subTable_row(self, header_name, data, index_name):
        self.check_sub_table(header_name)
        self.data[header_name][self.index].append(index_name)   
        for i,c in enumerate(self.columns):
            self.data[header_name][c].append( data[i] )
                   
    def get_table(self):
        dfs = []
        df_names = []
        for sub_table_header_name in self.data.keys():
            df = pd.DataFrame(self.data[sub_table_header_name]).set_index(self.index)
            dfs.append(df)
            df_names.append(sub_table_header_name)
        return pd.concat( dfs ,axis=1, keys= df_names)