import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Desktop\Kaggle_study\crx_data.txt', sep=",", header=None)
data.columns = ['A{}'.format(i) for i in range(1,17)]


#target 개수 차이
plt.figure(figsize=(15, 8))
at = data['A16'].value_counts().plot(kind='bar')

at.set(ylabel = 'count', title = 'compare count')
for p in range(len(at.patches)):
    at_patch = at.patches[p]
    at.text(at_patch.get_x()+at_patch.get_width()/2, at_patch.get_height()+10, data['A16'].value_counts()[p],
            ha = 'center')
plt.show()



#변수 종류 개수 
import seaborn as sns
plt.figure(figsize=(15, 8))
cols =  ['A{}'.format(i) for i in range(1,17) if data['A{}'.format(i)].dtype == 'O']
uniques = [len(data[col].unique()) for col in cols]
sns.set(font_scale=1.2)
pal = sns.color_palette()
ax = sns.barplot(cols, uniques, palette=pal)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 0.2,
            uniq,
            ha="center") 
            
            
#Numerical feature 변수 분포 확인            

sns.pairplot(data[cols + ['A16']], hue='A16', 
             x_vars=cols, y_vars=cols)
plt.show()            


#categorical 변수 별 target class 분포
cols =  ['A{}'.format(i) for i in range(1,16) if data['A{}'.format(i)].dtype == 'O']
for col in cols:
    
        
    data_df = data.groupby([col,'A16'])['A16'].count().unstack('A16')
    data_df.plot(kind='bar', figsize=(10,4))
    plt.title(col)
    plt.show()
            
  
#categorical 변수간 target class 분포
cols =  ['A{}'.format(i) for i in range(1,16) if data['A{}'.format(i)].dtype == 'O']
cols_2 =  ['A{}'.format(i) for i in range(1,16) if data['A{}'.format(i)].dtype == 'O']

for col in cols:
    for col_2 in cols_2:
        if col == col_2:
            continue
        else:
            data_df = data.groupby([col,col_2,'A16'])['A16'].count().unstack('A16')
            data_df.plot(kind='bar', figsize=(10,4))
            plt.title(col+col_2)
            plt.show()
    cols_2.remove(col)
  
  
  
  
  
