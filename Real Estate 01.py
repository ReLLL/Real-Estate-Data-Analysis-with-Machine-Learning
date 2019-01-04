#!/usr/bin/env python
# coding: utf-8

# # Real Estate Data analysis 
# (on the data scraped from sahibinden.com, for Antalya city)

# # Data Preprocessing

# In[1]:


import pandas as pd
import numpy as np
df = pd.read_csv('real_estate_antalya.csv', sep=',', encoding='utf8', decimal=',')


# In[2]:


df['fiyat'] = df['fiyat'].str.replace('.', '', regex=False)
df['fiyat'] = df['fiyat'].str.replace('TL', '', regex=False)
df['fiyat'] = pd.to_numeric(df['fiyat'], errors='coerce')
df['aidat_tl'] = df['aidat_tl'].str.replace('Belirtilmemiş', '', regex=False)
#df = df.replace(r'\s+', np.nan, regex=True)
df = df.replace('Hayır', 0, regex=False)
df = df.replace('Evet', 1, regex=False)
df = df.replace('Belirtilmemiş', np.nan, regex=False)

df = df.applymap(lambda x: 1 if x == True else x)
df = df.applymap(lambda x: 0 if x == False else x)
print(df.columns)


# In[3]:


# Transpose to easily see example values for each column 
print(df.T.iloc[:, :3])


# In[4]:


df['isitma'].unique()


# In[5]:


df['isitma_klima'] = df['isitma'] == 'Klima'
df['isitma_dogalgaz'] = df['isitma'] == 'Doğalgaz'
df['isitma_soba'] = df['isitma'] == 'Soba'
df['isitma_merkezi'] = df['isitma'] == 'Merkezi'
df['isitma_yok'] = df['isitma'] == 'Yok'
df.drop(['isitma'], axis=1, inplace=True)


# In[6]:


df['kat_sayisi'] = pd.to_numeric(df['kat_sayisi'], errors='coerce')
df['brut_m2'] = pd.to_numeric(df['brut_m2'], errors='coerce')


# In[7]:


df['oda_sayisi'].unique()


# In[8]:


df['salon'] = 0
df.loc[df.oda_sayisi.isin(['6+1', '5+1', '4+1', '3+1', '2+1', '1+1', '1.5+1', '2', '3', '4', '5', '6','7', '8']), 'salon'] = 1
df.loc[df.oda_sayisi.isin(['9+2', '8+2', '7+2', '6+2', '5+2', '4+2', '3+2']), 'salon'] = 2
df.loc[df.oda_sayisi.isin(['9+3', '8+3', '7+3', '6+3', '5+3']), 'salon'] = 3

df['oda'] = 0
df.loc[df.oda_sayisi.isin(['1+1', '1.5+1', '1', '2']), 'oda'] = 1
df.loc[df.oda_sayisi.isin(['2+1' '2+2', '3']), 'oda'] = 2
df.loc[df.oda_sayisi.isin(['3+1', '3+2', '4']), 'oda'] = 3
df.loc[df.oda_sayisi.isin(['4+1', '4+2', '5']), 'oda'] = 4
df.loc[df.oda_sayisi.isin(['5+1', '5+2', '5+3', '6']), 'oda'] = 5
df.loc[df.oda_sayisi.isin(['6+1', '6+2', '6+3', '7']), 'oda'] = 6
df.loc[df.oda_sayisi.isin(['7+1', '7+2', '7+3', '8']), 'oda'] = 7
df.loc[df.oda_sayisi.isin(['8+1', '8+2', '8+3', '9']), 'oda'] = 8

df['toplam_oda'] = 0
df.loc[df.oda_sayisi.isin(['1']), 'toplam_oda'] = 1
df.loc[df.oda_sayisi.isin(['2', '1+1', '1.5+1']), 'toplam_oda'] = 2
df.loc[df.oda_sayisi.isin(['3', '2+1']), 'toplam_oda'] = 3
df.loc[df.oda_sayisi.isin(['4', '3+1', '2+2']), 'toplam_oda'] = 4
df.loc[df.oda_sayisi.isin(['5', '4+1', '3+2']), 'toplam_oda'] = 5
df.loc[df.oda_sayisi.isin(['6', '5+1', '4+2']), 'toplam_oda'] = 6
df.loc[df.oda_sayisi.isin(['7', '6+1', '5+2', '4+3']), 'toplam_oda'] = 7
df.loc[df.oda_sayisi.isin(['8', '7+1', '6+2', '5+3']), 'toplam_oda'] = 8
df.loc[df.oda_sayisi.isin(['9', '8+1', '7+2', '6+3']), 'toplam_oda'] = 9
df.loc[df.oda_sayisi.isin(['10', '9+1', '8+2', '7+3', '6+4']), 'toplam_oda'] = 10

print(df['salon'].unique())
print(df['oda'].unique())
print(df['toplam_oda'].unique())


# In[9]:


df['kullanim_durumu'].unique()


# In[10]:


df['kullanim_durumu'] = df['kullanim_durumu'] == 'Kiracılı'


# In[11]:


df['semt'].unique() 


# In[12]:


dummy = pd.get_dummies(df['semt'], prefix='semt')
dummy.head()


# In[13]:


df = pd.concat([df, dummy], axis=1)
df.head()


# In[14]:


df['mahalle'].unique() 


# In[15]:


dummy = pd.get_dummies(df['mahalle'], prefix='mahalle')
dummy.head()


# In[16]:


df = pd.concat([df, dummy], axis=1)
df.head()


# In[17]:


df['bina_yasi'].unique()


# In[18]:


df['yas'] = np.NaN
for n in range(0, 100):
    df.loc[df.bina_yasi == str(n), 'yas'] = n
df.loc[df.bina_yasi == '5-10 arası', 'yas'] = 7
df.loc[df.bina_yasi == '11-15 arası', 'yas'] = 13
df.loc[df.bina_yasi == '16-20 arası', 'yas'] = 18
df.loc[df.bina_yasi == '21-25 arası', 'yas'] = 23
df.loc[df.bina_yasi == '26-30 arası', 'yas'] = 28
df.loc[df.bina_yasi == '31-35 arası', 'yas'] = 33
df.loc[df.bina_yasi == '36-40 arası', 'yas'] = 38
df.loc[df.bina_yasi == '41-45 arası', 'yas'] = 43
df.loc[df.bina_yasi == '46-50 arası', 'yas'] = 48
df.loc[df.bina_yasi == '51-55 arası', 'yas'] = 53
df.loc[df.bina_yasi == '56-60 arası', 'yas'] = 58
df.loc[df.bina_yasi == '61-65 arası', 'yas'] = 63
df.loc[df.bina_yasi == '66-70 arası', 'yas'] = 68
df.loc[df.bina_yasi == '71-75 arası', 'yas'] = 73
df.loc[df.bina_yasi == '76-80 arası', 'yas'] = 78
df.loc[df.bina_yasi == '81-85 arası', 'yas'] = 83
df.loc[df.bina_yasi == '86-90 arası', 'yas'] = 88
df['yas'].unique()


# In[19]:


df['bulundugu_kat'].unique()


# In[20]:


dummy = pd.get_dummies(df['bulundugu_kat'], prefix='kat')
dummy.head()


# In[21]:


df = pd.concat([df, dummy], axis=1)
df.head()


# In[22]:


df['kat'] = np.NaN
for n in range(0, 30):
    df.loc[df.bulundugu_kat == str(n), 'kat'] = n
df.loc[df.bulundugu_kat == 'Yüksek Giriş', 'kat'] = 0.5
df.loc[df.bulundugu_kat == 'Giriş Katı', 'kat'] = 0
df.loc[df.bulundugu_kat == 'Çatı Katı', 'kat'] = 5
df.loc[df.bulundugu_kat == 'Kot 2', 'kat'] = -2
df.loc[df.bulundugu_kat == 'Kot 1', 'kat'] = -1
df.loc[df.bulundugu_kat == 'Kot 3', 'kat'] = -3
df.loc[df.bulundugu_kat == 'Müstakil', 'kat'] = 0
df.loc[df.bulundugu_kat == 'Bahçe Katı', 'kat'] = 0
df.loc[df.bulundugu_kat == 'Zemin Kat', 'kat'] = 0
df.loc[df.bulundugu_kat == 'Villa Tipi', 'kat'] = 1
df.loc[df.bulundugu_kat == '30 ve üzeri', 'kat'] = 30


# In[23]:


df['emlak_tipi'].unique()


# # Data Exploration

# In[24]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


print(df.groupby('emlak_tipi').size())


# In[26]:


#sns.pairplot(df['emlak_tipi', 'fiyat', 'kat', 'yas'], hue='emlak_tipi', vars=['fiyat', 'kat', 'yas'], height=3, aspect=1)
d = df[['semt',  'kat_sayisi', 'yas', 'toplam_oda']]
d = d[d.semt.isin(['Kepez', 'Aksu', 'Muratpaşa', 'Konyaaltı', 'Döşemealtı', 'Serik'])]
d.dropna(inplace=True)
d.head()


# ![350px-Antalya_districts%5B1%5D.png](attachment:350px-Antalya_districts%5B1%5D.png)

# In[27]:


sns.pairplot(d, hue='semt', height=5, aspect=1)
plt.show()


# In[28]:


print('All Real Estate Objects:', len(df))
df = df[df.emlak_tipi == 'Satılık Daire']
len(df)


# In[29]:


d.hist(edgecolor='black', linewidth=1.2,  figsize=(12, 8))
plt.show()


# In[30]:


d = df.isnull().sum()
d[d >0]


# In[31]:


d = df.loc[df.fiyat < 500000, 'fiyat']
from scipy.stats import norm
sns.distplot(d, bins=30,  fit=norm)
plt.show()


# In[32]:


cols = ['fiyat', 'semt',  'kat_sayisi', 'yas', 'toplam_oda', 'bulundugu_kat', 'yuzme_havuzu',
        'asansor', 'balkon','boyali', 'deniz_manzarali', 'brut_m2', 'guney', 'guvenlik',
        'site_icerisinde', 'laminat_zemin', 'parke_zemin', 'teras']
d = df[cols]
d.corr()


# In[33]:


pd.options.display.float_format = '{:,.2f}'.format  # Without this, it looks cluttured, too many decimal places
plt.figure(figsize=(16,10))
#sns.heatmap(df.corr(), annot=True, cmap="RdYlBu_r")
sns.heatmap(d.corr(), annot=True, cmap="coolwarm")
plt.show()


# In[34]:


cols = ['fiyat']
cols.extend([x for x in df.columns if 'semt_' in x])
d = df[cols]

plt.figure(figsize=(16,10))
#sns.heatmap(df.corr(), annot=True, cmap="RdYlBu_r")
sns.heatmap(d.corr(), annot=True, cmap="coolwarm")
plt.show()


# In[35]:


print(len(df.mahalle.unique()))
d = df[df.semt=='Kepez']
print(len(d.mahalle.unique()))
mahs = ['mahalle_' + x for x in d.mahalle.unique()]
print(mahs)
cols = ['fiyat']
#cols.extend([x for x in d.columns if 'mahalle_' in x])
cols.extend(mahs)
d = d[cols]
d.corr()


# In[36]:


d = d.corr()
d = d[d.abs() > 0.2]
s = d.unstack()
so = s.sort_values(kind="quicksort")
print(so)


# In[37]:


d = df.loc[df.fiyat < 500000]
grid = sns.JointGrid(x='toplam_oda', y='fiyat', data=d, space=0, height=6, ratio=50)
grid.plot_joint(sns.regplot, color="b", order=2)
grid.plot_marginals(sns.rugplot, color="b", height=4)
plt.show()


# In[38]:


d = df[['fiyat', 'toplam_oda', 'yas', 'kat']]
d.dropna(inplace=True)
X = d.drop(['fiyat'], axis=1)
y = d['fiyat'].values
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y.reshape(-1,1)).flatten()
model = LinearRegression()
model.fit(X_std, y_std)
model.coef_


# In[39]:


print(df.groupby('semt').size())


# ### Apartments cheaper than 130.000 TL
# Ordered by average square meter price

# In[40]:


d = df[df.fiyat < 130000]
d = d[['fiyat', 'semt', 'mahalle', 'toplam_oda', 'kat', 'yas', 'asansor', 'brut_m2']]
d['unit_m2'] = d['fiyat'] / d['brut_m2']
d.sort_values(by=['unit_m2'], inplace=True, ascending=False)
d


# ### Average price of square meter for each neighboorhood
# (just for the objects below 500.000 TL price and above ground floor)

# In[41]:


d = df[df.fiyat < 500000]
d = d[df.kat >= 1.0]
d['unit_m2'] = d['fiyat'] / d['brut_m2']
d = d[['mahalle', 'unit_m2']]
d = d.groupby(['mahalle']).mean()
d.sort_values(by=['unit_m2'], inplace=True, ascending=False)
d

