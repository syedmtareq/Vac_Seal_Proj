#!/usr/bin/env python
# coding: utf-8

# In[56]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

r2 = 7.132
e = .05
v = .2
eprime = e
t = 2.14

skip = np.arange(0,1)
headers = ['time', 'extension', 'load']
df = pd.read_csv('D:\\Yang\\Vac_Seal_Proj\\StrengthAna\\vacsealproj_strengthrawdata\\A7_8__T_S10_ST_1_1.csv',names = headers, skiprows=skip)

from scipy.interpolate import interp1d

time1 = np.arange(0,16)
d1 = interp1d(df['time'],df['extension'])
d = d1(time1)

f1 = interp1d(df['time'], df['load'])
P = f1(time1)
M = (P/4/np.pi)*((1+v)*np.log(r2/2/eprime)+1)
E = 1.4177*(10**-6)*(320**3.25)/1000
D = (E*(t**3))/(12*(1-(v**2)))

stress = 6*M/(t**2)
dmax = ((3+v)*P.max()*((r2/2**2)))/(16*np.pi*(1+v)*D)*1000
strain_per = d[0:13]/dmax*100

plt.plot(strain_per, stress[0:13], '-', label = '1 Day', color = 'blue')

skip = np.arange(0,1)
headers = ['time', 'extension', 'load']
df = pd.read_csv('D:\\Yang\\Vac_Seal_Proj\\StrengthAna\\vacsealproj_strengthrawdata\\A3_1__T_S10_ST_2.csv',names = headers, skiprows=skip)


time1 = np.arange(0,34)
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
P = f1(time1)
M = (P/4/np.pi)*((1+v)*np.log(r2/2/eprime)+1)
E = 1.4177*(10**-6)*(320**3.25)/1000
D = (E*(t**3))/(12*(1-(v**2)))

stress = 6*M/(t**2)
dmax = ((3+v)*P.max()*((r2/2**2)))/(16*np.pi*(1+v)*D)*1000

strain_per = d[0:30]/dmax*100

plt.plot(strain_per, stress[0:30], '-', color = 'red', label='3 Days', lw=1)


r2 = 7.132
e = .05
v = .2
eprime = e
t = 2.14

skip = np.arange(0,1)
headers = ['time', 'extension', 'load']
df = pd.read_csv('D:\\Yang\\Vac_Seal_Proj\\StrengthAna\\vacsealproj_strengthrawdata\\A7_3__T_S10_ST_1.csv',names = headers, skiprows=skip)

from scipy.interpolate import interp1d

time1 = np.arange(0,26)
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
P = f1(time1)
M = (P/4/np.pi)*((1+v)*np.log(r2/2/eprime)+1)
E = 1.4177*(10**-6)*(320**3.25)/1000
D = (E*(t**3))/(12*(1-(v**2)))

stress = 6*M/(t**2)
dmax = ((3+v)*P.max()*((r2/2**2)))/(16*np.pi*(1+v)*D)*1000

strain_per = d/dmax*100

plt.plot(strain_per, stress, '-', color = 'g',label='7 Days')
plt.legend(loc='best')
plt.title('Aging Strengths', fontdict={'fontsize': 20})
plt.xlabel('% Strain',fontdict={'fontsize': 20})
plt.ylabel('Flexural Stress (MPa)',fontdict={'fontsize': 20})
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

skip = np.arange(0,1)
headers = ['time', 'extension', 'load']
df = pd.read_csv('D:\\Yang\\Vac_Seal_Proj\\StrengthAna\\vacsealproj_strengthrawdata\\A7_8__T_S10_ST_1_1.csv',names = headers, skiprows=skip)

from scipy.interpolate import interp1d

time1 = np.arange(0,16)
L = 7.5
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)
R = 3.75

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
F = f1(time1)
D = d1(F.argmax())

stress = (F*L)/(np.pi*(R**3))
strain = (6*D*d/(L**2))

strain_per = strain/(6*D*D/(L**2))*100

plt.plot(strain_per, stress, '-', color = 'blue')


# In[ ]:





# In[25]:


skip = np.arange(0,1)
headers = ['time', 'extension', 'load']
df = pd.read_csv('D:\\Yang\\Vac_Seal_Proj\\StrengthAna\\vacsealproj_strengthrawdata\\A3_1__T_S10_ST_2.csv',names = headers, skiprows=skip)


time1 = np.arange(0,34)
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
P = f1(time1)
M = (P/4/np.pi)*((1+v)*np.log(r2/2/eprime)+1)
E = 1.4177*(10**-6)*(320**3.25)/1000
D = (E*(t**3))/(12*(1-(v**2)))

stress = 6*M/(t**2)
dmax = ((3+v)*P.max()*((r2/2**2)))/(16*np.pi*(1+v)*D)*1000

strain_per = d[0:30]/dmax*100

plt.plot(strain_per, stress[0:30], '-', color = 'red', label='sealed 10s, A3_1, 2', lw=1)
plt.legend(loc='best')
#plt.xlim(0,100)
plt.title('A3')
plt.xlabel('% Strain')
plt.ylabel('Flexural Stress')
plt.show()


# In[ ]:





# In[37]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

r2 = 7.132
e = .05
v = .2
eprime = e
t = 2.14

skip = np.arange(0,1)
headers = ['time', 'extension', 'load']
df = pd.read_csv('D:\\Yang\\Vac_Seal_Proj\\StrengthAna\\vacsealproj_strengthrawdata\\A7_3__T_S10_ST_1.csv',names = headers, skiprows=skip)

from scipy.interpolate import interp1d

time1 = np.arange(0,26)
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
P = f1(time1)
M = (P/4/np.pi)*((1+v)*np.log(r2/2/eprime)+1)
E = 1.4177*(10**-6)*(320**3.25)/1000
D = (E*(t**3))/(12*(1-(v**2)))

stress = 6*M/(t**2)
dmax = ((3+v)*P.max()*((r2/2**2)))/(16*np.pi*(1+v)*D)*1000

strain_per = d/dmax*100

plt.plot(strain_per, stress, '-', color = 'blue',label='sealed 10s', lw=1)




# In[ ]:





# In[ ]:


plt.legend(loc='best')
plt.xlim(0, 100)
plt.title('Aging Strengths')
plt.xlabel('% Strain')
plt.ylabel('Flexural Stress')
plt.show()


# In[ ]:





# In[ ]:





# In[12]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

skip = np.arange(0,1)
headers = ['time', 'extension', 'load']
df = pd.read_csv('D:\\Yang\\Vac_Seal_Proj\\StrengthAna\\vacsealproj_strengthrawdata\\A3_2__T_S10_ST_WithoutSeal.csv',names = headers, skiprows=skip)

from scipy.interpolate import interp1d

time1 = np.arange(0,26)
L = 7.5
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)
R = 3.75

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
F = f1(time1)
D = d1(F.argmax())

stress = (F*L)/(np.pi*(R**3))
strain = (6*D*d/(L**2))

strain_per = strain/(6*D*D/(L**2))*100

# plt.plot(strain_per, stress, '-', color = 'blue')

plt.plot(strain_per, stress, '-', color = 'g',label='7 Days')
plt.legend(loc='best')
plt.title('Aging Strengths', fontdict={'fontsize': 10})
plt.xlabel('% Strain',fontdict={'fontsize': 10})
plt.ylabel('Flexural Stress (MPa)',fontdict={'fontsize': 10})
plt.show()


# In[ ]:





# In[ ]:





# In[16]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

skip = np.arange(0,1)
headers = ['time', 'extension', 'load']
df = pd.read_csv('D:\\Yang\\Vac_Seal_Proj\\StrengthAna\\vacsealproj_strengthrawdata\\A3_1__T_S10_ST_WithoutSeal.csv',names = headers, skiprows=skip)

from scipy.interpolate import interp1d

time1 = np.arange(0,16)
L = 7.5
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)
R = 3.75

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
F = f1(time1)
D = d1(F.argmax())

stress = (F*L)/(np.pi*(R**3))
strain = (6*D*d/(L**2))

strain_per = strain/(6*D*D/(L**2))*100

plt.plot(strain_per, stress, '-', color = 'blue')



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

skip = np.arange(0,1)
headers = ['time', 'extension', 'load']
df = pd.read_csv('C:\\Users\\User\\Desktop\\A1_1__T_S10_ST_1.csv',names = headers, skiprows=skip)

from scipy.interpolate import interp1d

time1 = np.arange(0,120)
L = 7.5
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)
R = 3.75

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
F = f1(time1)
D = d1(F.argmax())

stress = (F*L)/(np.pi*(R**3))
strain = (6*D*d/(L**2))

strain_per = strain/(6*D*D/(L**2))*100

plt.plot(strain_per, stress, '-', color = 'red', label='A1', lw=1)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

skip = np.arange(0,1)
headers = ['time', 'extension', 'load']
df = pd.read_csv('C:\\Users\\User\\Desktop\\A7_1__T_S10_ST_1.csv',names = headers, skiprows=skip)

from scipy.interpolate import interp1d

time1 = np.arange(0,26)
L = 7.5
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)
R = 3.75

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
F = f1(time1)
D = d1(F.argmax())

stress = (F*L)/(np.pi*(R**3))
strain = (6*D*d/(L**2))

strain_per = strain/(6*D*D/(L**2))*100

plt.plot(strain_per, stress, '-', color = 'blue')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

skip = np.arange(0,1)
headers = ['time', 'extension', 'load']
df = pd.read_csv('C:\\Users\\User\\Desktop\\A7_4_T_S10_ST_1.csv',names = headers, skiprows=skip)

from scipy.interpolate import interp1d

time1 = np.arange(0,17)
L = 7.5
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)
R = 3.75

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
F = f1(time1)
D = d1(F.argmax())

stress = (F*L)/(np.pi*(R**3))
strain = (6*D*d/(L**2))

strain_per = strain/(6*D*D/(L**2))*100

plt.plot(strain_per, stress, '-', color = 'blue')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

skip = np.arange(0,1)
headers = ['time', 'extension', 'load']
df = pd.read_csv('C:\\Users\\User\\Desktop\\A7_7__T_S10_ST_1.csv',names = headers, skiprows=skip)

from scipy.interpolate import interp1d

time1 = np.arange(0,28)
L = 7.5
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)
R = 3.75

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
F = f1(time1)
D = d1(F.argmax())

stress = (F*L)/(np.pi*(R**3))
strain = (6*D*d/(L**2))

strain_per = strain/(6*D*D/(L**2))*100

plt.plot(strain_per, stress, '-', color = 'blue')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

skip = np.arange(0,1)
headers = ['time', 'extension', 'load']
df = pd.read_csv('C:\\Users\\User\\Desktop\\A3_2__T_S10_ST_1.csv',names = headers, skiprows=skip)

from scipy.interpolate import interp1d

time1 = np.arange(0,21)
L = 7.5
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)
R = 3.75

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
F = f1(time1)
D = d1(F.argmax())

stress = (F*L)/(np.pi*(R**3))
strain = (6*D*d/(L**2))

strain_per = strain/(6*D*D/(L**2))*100

plt.plot(strain_per, stress, '-', color = 'orange', label='A3',lw=1)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

skip = np.arange(0,1)
headers = ['time', 'extension', 'load']
df = pd.read_csv('C:\\Users\\User\\Desktop\\A7_9__T_S10_ST_2.csv',names = headers, skiprows=skip)

from scipy.interpolate import interp1d

time1 = np.arange(0,36)
L = 7.5
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)
R = 3.75

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
F = f1(time1)
D = d1(F.argmax())

stress = (F*L)/(np.pi*(R**3))
strain = (6*D*d/(L**2))

strain_per = strain/(6*D*D/(L**2))*100

plt.plot(strain_per, stress, '-', color = 'blue')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

skip = np.arange(0,1)
headers = ['time', 'extension', 'load']
df = pd.read_csv('C:\\Users\\User\\Desktop\\A7_6__T_S10_ST_1.csv',names = headers, skiprows=skip)

from scipy.interpolate import interp1d

time1 = np.arange(0,19)
L = 7.5
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)
R = 3.75

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
F = f1(time1)
D = d1(F.argmax())

stress = (F*L)/(np.pi*(R**3))
strain = (6*D*d/(L**2))

strain_per = strain/(6*D*D/(L**2))*100

plt.plot(strain_per, stress, '-', color = 'blue')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

skip = np.arange(0,1)
headers = ['time', 'extension', 'load']
df = pd.read_csv('C:\\Users\\User\\Desktop\\A7_3__T_S10_ST_1.csv',names = headers, skiprows=skip)

from scipy.interpolate import interp1d

time1 = np.arange(0,26)
L = 7.5
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)
R = 3.75

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
F = f1(time1)
D = d1(F.argmax())

stress = (F*L)/(np.pi*(R**3))
strain = (6*D*d/(L**2))

strain_per = strain/(6*D*D/(L**2))*100

plt.plot(strain_per, stress, '-', color = 'blue')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

skip = np.arange(0,1)
headers = ['time', 'extension', 'load']
df = pd.read_csv('C:\\Users\\User\\Desktop\\A7_2__T_S10_ST_1.csv',names = headers, skiprows=skip)

from scipy.interpolate import interp1d

time1 = np.arange(0,39)
L = 7.5
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)
R = 3.75

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
F = f1(time1)
D = d1(F.argmax())

stress = (F*L)/(np.pi*(R**3))
strain = (6*D*d/(L**2))

strain_per = strain/(6*D*D/(L**2))*100

plt.plot(strain_per, stress, '-', color = 'blue')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

skip = np.arange(0,1)
headers = ['time', 'extension', 'load']
df = pd.read_csv('C:\\Users\\User\\Desktop\\A3_1__T_S10_ST_1.csv',names = headers, skiprows=skip)

from scipy.interpolate import interp1d

time1 = np.arange(0,232)
L = 7.5
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)
R = 3.75

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
F = f1(time1)
D = d1(F.argmax())

stress = (F*L)/(np.pi*(R**3))
strain = (6*D*d/(L**2))

strain_per = strain/(6*D*D/(L**2))*100

plt.plot(strain_per, stress, '-', color = 'orange')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

skip = np.arange(0)
headers = ['time', 'extension', 'load']
df = pd.read_csv('C:\\Users\\User\\Desktop\\A7_9__T_S10_ST_1.csv',names = headers, skiprows=skip)

from scipy.interpolate import interp1d

time1 = np.arange(0,55)
L = 7.5
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)
R = 3.75

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
F = f1(time1)
D = d1(F.argmax())

stress = (F*L)/(np.pi*(R**3))
strain = (6*D*d/(L**2))

strain_per = strain/(6*D*D/(L**2))*100

plt.plot(strain_per, stress, '-', color = 'blue',label='A7', lw=1)
plt.legend(loc='best')
plt.xlim(0, 100)
plt.title('Aging Strengths')
plt.xlabel('% Strain')
plt.ylabel('Flexural Stress')
plt.show()


# In[ ]:





# In[24]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

r2 = 7.132
e = .05
v = .2
eprime = e
t = 2.14

skip = np.arange(0,1)
headers = ['time', 'extension', 'load']
df = pd.read_csv('D:\\Yang\\Vac_Seal_Proj\\StrengthAna\\vacsealproj_strengthrawdata\\A3_2__T_S10_ST_WithoutSeal.csv',names = headers, skiprows=skip)

from scipy.interpolate import interp1d

time1 = np.arange(0,22)
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
P = f1(time1)
M = (P/4/np.pi)*((1+v)*np.log(r2/2/eprime)+1)
E = 1.4177*(10**-6)*(320**3.25)/1000
D = (E*(t**3))/(12*(1-(v**2)))

stress = 6*M/(t**2)
dmax = ((3+v)*P.max()*((r2/2**2)))/(16*np.pi*(1+v)*D)*1000

strain_per = d/dmax*100

plt.plot(strain_per, stress, '-', color = 'black',label='no seal, 1', lw=1)



skip = np.arange(0,1)
headers = ['time', 'extension', 'load']
df = pd.read_csv('D:\\Yang\\Vac_Seal_Proj\\StrengthAna\\vacsealproj_strengthrawdata\\A3_2__T_S10_ST_1.csv',names = headers, skiprows=skip)

from scipy.interpolate import interp1d

time1 = np.arange(0,21)
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
P = f1(time1)
M = (P/4/np.pi)*((1+v)*np.log(r2/2/eprime)+1)
E = 1.4177*(10**-6)*(320**3.25)/1000
D = (E*(t**3))/(12*(1-(v**2)))

stress = 6*M/(t**2)
dmax = ((3+v)*P.max()*((r2/2**2)))/(16*np.pi*(1+v)*D)*1000

strain_per = d/dmax*100

plt.plot(strain_per, stress, '-', color = 'blue',label='sealed 10s, A3_2', lw=1)



skip = np.arange(0,1)
headers = ['time', 'extension', 'load']
df = pd.read_csv('D:\\Yang\\Vac_Seal_Proj\\StrengthAna\\vacsealproj_strengthrawdata\\A3_1__T_S10_ST_WithoutSeal.csv',names = headers, skiprows=skip)

from scipy.interpolate import interp1d

time1 = np.arange(0,120)
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
P = f1(time1)
M = (P/4/np.pi)*((1+v)*np.log(r2/2/eprime)+1)
E = 1.4177*(10**-6)*(320**3.25)/1000
D = (E*(t**3))/(12*(1-(v**2)))

stress = 6*M/(t**2)
dmax = ((3+v)*P.max()*((r2/2**2)))/(16*np.pi*(1+v)*D)*1000

strain_per = d[0:60]/dmax*100

plt.plot(strain_per, stress[0:60], '-', color = 'orange', label='No seal, 2', lw=1)





##skip = np.arange(0,1)
##headers = ['time', 'extension', 'load']
##df = pd.read_csv('C:\\Users\\mcols\\Desktop\\A3_1__T_S10_ST_1.csv',names = headers, skiprows=skip)
##
##
##time1 = np.arange(0,232)
##d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
##d = d1(time1)
##
##f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
##P = f1(time1)
##M = (P/4/np.pi)*((1+v)*np.log(r2/2/eprime)+1)
##E = (P.max())/(d.max())
##D = (E*(t**3))/(12*(1-(v**2)))
##
##stress = 6*M/(t**2)
##dmax = ((3+v)*P.max()*((r2/2**2)))/(16*np.pi*(1+v)*D)*1000
##
##strain_per = d/dmax*100
##
##plt.plot(strain_per, stress, '-', color = 'green', label='sealed 10s, A3_1, 1', lw = 1)
##


skip = np.arange(0,1)
headers = ['time', 'extension', 'load']
df = pd.read_csv('D:\\Yang\\Vac_Seal_Proj\\StrengthAna\\vacsealproj_strengthrawdata\\A3_1__T_S10_ST_2.csv',names = headers, skiprows=skip)


time1 = np.arange(0,34)
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
P = f1(time1)
M = (P/4/np.pi)*((1+v)*np.log(r2/2/eprime)+1)
E = 1.4177*(10**-6)*(320**3.25)/1000
D = (E*(t**3))/(12*(1-(v**2)))

stress = 6*M/(t**2)
dmax = ((3+v)*P.max()*((r2/2**2)))/(16*np.pi*(1+v)*D)*1000

strain_per = d[0:30]/dmax*100

plt.plot(strain_per, stress[0:30], '-', color = 'red', label='sealed 10s, A3_1, 2', lw=1)
plt.legend(loc='best')
#plt.xlim(0,100)
plt.title('A3')
plt.xlabel('% Strain')
plt.ylabel('Flexural Stress')
plt.show()


# In[ ]:





# In[29]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

r2 = 7.132
e = .05
v = .2
eprime = e
t = 2.14
skip = np.arange(0)
headers = ['time', 'extension', 'load']
df = pd.read_csv('D:\\Yang\\Vac_Seal_Proj\\StrengthAna\\vacsealproj_strengthrawdata\\A7_9__T_S10_ST_1.csv',names = headers, skiprows=skip)

from scipy.interpolate import interp1d
time1 = np.arange(0,55)
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
P = f1(time1)
M = (P/4/np.pi)*((1+v)*np.log(r2/2/eprime)+1)
E = 1.4177*(10**-6)*(320**3.25)/1000
D = (E*(t**3))/(12*(1-(v**2)))

stress = 6*M/(t**2)
dmax = ((3+v)*P.max()*((r2/2**2)))/(16*np.pi*(1+v)*D)*1000

strain_per = d/dmax*100

plt.plot(strain_per, stress, '-', color = 'blue',label='sealed 10s, 2', lw=1)



skip = np.arange(0,8)
headers = ['time', 'extension', 'load']
df = pd.read_csv('D:\\Yang\\Vac_Seal_Proj\\StrengthAna\\vacsealproj_strengthrawdata\\A7_9__T_S10_ST_WithoutSeal.csv',names = headers, skiprows=skip)

from scipy.interpolate import interp1d

time1 = np.arange(0,63)
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
P = f1(time1)
M = (P/4/np.pi)*((1+v)*np.log(r2/2/eprime)+1)
E = 1.4177*(10**-6)*(320**3.25)/1000
D = (E*(t**3))/(12*(1-(v**2)))

stress = 6*M/(t**2)
dmax = ((3+v)*P.max()*((r2/2**2)))/(16*np.pi*(1+v)*D)*1000

strain_per = d[0:60]/dmax*100

plt.plot(strain_per, stress[0:60], '-', color = 'black',label='no seal', lw=1)



skip = np.arange(0,1)
headers = ['time', 'extension', 'load']
df = pd.read_csv('D:\\Yang\\Vac_Seal_Proj\\StrengthAna\\vacsealproj_strengthrawdata\\A7_9__T_S10_ST_2.csv',names = headers, skiprows=skip)

from scipy.interpolate import interp1d

time1 = np.arange(0,36)
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
P = f1(time1)
M = (P/4/np.pi)*((1+v)*np.log(r2/2/eprime)+1)
E = 1.4177*(10**-6)*(320**3.25)/1000
D = (E*(t**3))/(12*(1-(v**2)))

stress = 6*M/(t**2)
dmax = ((3+v)*P.max()*((r2/2**2)))/(16*np.pi*(1+v)*D)*1000

strain_per = d/dmax*100

plt.plot(strain_per, stress, '-', color = 'red',label='sealed 10s, 1',lw=1)
plt.legend(loc='best')
#plt.xlim(0,100)
plt.title('A7_9')
plt.xlabel('% Strain')
plt.ylabel('Flexural Stress')
plt.show()


# In[ ]:





# In[30]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

r2 = 7.132
e = .05
v = .2
eprime = e
t = 2.14
skip = np.arange(0)
headers = ['time', 'extension', 'load']
df = pd.read_csv('D:\\Yang\\Vac_Seal_Proj\\StrengthAna\\vacsealproj_strengthrawdata\\A7_9__T_S10_ST_1.csv',names = headers, skiprows=skip)

from scipy.interpolate import interp1d
time1 = np.arange(0,55)
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
P = f1(time1)
M = (P/4/np.pi)*((1+v)*np.log(r2/2/eprime)+1)
E = 1.4177*(10**-6)*(320**3.25)/1000
D = (E*(t**3))/(12*(1-(v**2)))

stress = 6*M/(t**2)
dmax = ((3+v)*P.max()*((r2/2**2)))/(16*np.pi*(1+v)*D)*1000

strain_per = d/dmax*100

plt.plot(strain_per, stress, '-', color = 'blue',label='sealed 10s, 2', lw=1)


# In[31]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

r2 = 7.132
e = .05
v = .2
eprime = e
t = 2.14
skip = np.arange(0)
headers = ['time', 'extension', 'load']
df = pd.read_csv('D:\\Yang\\Vac_Seal_Proj\\StrengthAna\\vacsealproj_strengthrawdata\\A7_9__T_S10_ST_1.csv',names = headers, skiprows=skip)
df


# In[32]:


from scipy.interpolate import interp1d
time1 = np.arange(0,55)
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
P = f1(time1)
M = (P/4/np.pi)*((1+v)*np.log(r2/2/eprime)+1)
E = 1.4177*(10**-6)*(320**3.25)/1000
D = (E*(t**3))/(12*(1-(v**2)))

stress = 6*M/(t**2)
dmax = ((3+v)*P.max()*((r2/2**2)))/(16*np.pi*(1+v)*D)*1000

strain_per = d/dmax*100

plt.plot(strain_per, stress, '-', color = 'blue',label='sealed 10s, 2', lw=1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[33]:


from scipy.interpolate import interp1d
time1 = np.arange(0,55)
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)


# In[ ]:





# In[ ]:





# In[34]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

r2 = 7.132
e = .05
v = .2
eprime = e
t = 2.14

skip = np.arange(0,1)
headers = ['time', 'extension', 'load']
df = pd.read_csv('D:\\Yang\\Vac_Seal_Proj\\StrengthAna\\vacsealproj_strengthrawdata\\A7_3__T_S10_ST_1.csv',names = headers, skiprows=skip)

from scipy.interpolate import interp1d

time1 = np.arange(0,26)
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
P = f1(time1)
M = (P/4/np.pi)*((1+v)*np.log(r2/2/eprime)+1)
E = 1.4177*(10**-6)*(320**3.25)/1000
D = (E*(t**3))/(12*(1-(v**2)))

stress = 6*M/(t**2)
dmax = ((3+v)*P.max()*((r2/2**2)))/(16*np.pi*(1+v)*D)*1000

strain_per = d/dmax*100

plt.plot(strain_per, stress, '-', color = 'blue',label='sealed 10s', lw=1)




skip = np.arange(0,1)
headers = ['time', 'extension', 'load']
df = pd.read_csv('D:\\Yang\\Vac_Seal_Proj\\StrengthAna\\vacsealproj_strengthrawdata\\A7_3__T_S10_ST_WithoutSeal.csv',names = headers, skiprows=skip)

from scipy.interpolate import interp1d

time1 = np.arange(0,41)
d1 = interp1d(df['time'],df['extension'], fill_value = 'extrapolate')
d = d1(time1)

f1 = interp1d(df['time'],df['load'], fill_value = 'extrapolate')
P = f1(time1)
M = (P/4/np.pi)*((1+v)*np.log(r2/2/eprime)+1)
E = 1.4177*(10**-6)*(320**3.25)/1000
D = (E*(t**3))/(12*(1-(v**2)))

stress = 6*M/(t**2)
dmax = ((3+v)*P.max()*((r2/2**2)))/(16*np.pi*(1+v)*D)*1000

strain_per = d/dmax*100

plt.plot(strain_per, stress, '-', color = 'black', label='no seal', lw=1)
plt.legend(loc='best')
#plt.xlim(0,100)
plt.title('A7_3')
plt.xlabel('% Strain')
plt.ylabel('Flexural Stress')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




