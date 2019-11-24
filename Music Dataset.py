
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.optimize import curve_fit
get_ipython().magic(u'matplotlib inline')


#### Import Data File

# In[2]:

from pandas import read_csv
data_load = read_csv('barry_recon2.csv')


# In[3]:

#print data_load


# In[4]:

#dt=np.dtype([('time',int,),
#             ('lat','a5'),
#             ('lon','a5'),
#             ('stat',int),
#             ('geo',int),
#             ('pres',int),
#             ('temp','a5'),
#             ('dew','a5'),
#             ('wind',int),
#             ('peak',int),
#             ('sfc',int),
#             ('rain',int),
#             ('flag',int)])


# In[5]:

data=np.loadtxt('barry_recon2.csv',delimiter=',',skiprows=1)
#result['price']


# In[6]:

#data = np.loadtxt('barry_data1.csv',delimiter=',',skiprows=0) 
#data = np.loadtxt('barry_recon.dat') 


# In[7]:

print data[:,9]


# In[8]:

var1 = data[:,5]
coffee = data[:,9]
#var3 = data[:,2]

data_len = len(var1)


# In[9]:

var2 = coffee/2.
print var2
#print coffee


# In[10]:

print var2.min()


# In[11]:

#plt.plot(var1, 'r--', var2, 'g--', var3, 'b--')
plt.plot(var1, 'r--', var2, 'g--')


# In[267]:

import matplotlib.animation as animation
#plt.rcParams['animation.ffmpeg_path'] = '/Users/samberkseth/Downloads/ffmpeg'
fig, ax = plt.subplots()
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
#Writer = animation.FFMpegWriter(fps=30, codec='libx264')  #or 
#Writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='Me'))
#Writer = animation.writers['mencoder']
#writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

t= np.arange(0,242,1)
ax.set_xlabel('time')
ax.set_ylabel('Pressure (mb) ', color='r')
ax.plot(t,var1,'r')
ax.tick_params(axis='y', colors='r')
ax.axes.get_xaxis().set_visible(False)

ax2 = ax.twinx()
ax2.set_ylabel('Peak 10s Wind (kt)', color='b')  # we already handled the x-label with ax1
ax2.plot(t,var2, 'b')
ax2.tick_params(axis='y', colors='b')
plt.axvline(    x=242,color='g')
plt.savefig("barry242.png")


### 

# In[ ]:




#### Finding the Variances from Point to Point (Changes over Time)

####### Temperature

# In[11]:

diff1 = np.zeros(data_len)

for x in range(data_len):
    diff1[x] = var1[x] - var1[x-1]

var1_diff = diff1[1:]
abs_var1 = np.abs(var1_diff)
print abs_var1


####### Dewpoint

# In[12]:

diff2 = np.zeros(data_len)

for x in range(data_len):
    diff2[x] = var2[x] - var2[x-1]
    
var2_diff = diff2[1:]
abs_var2 = np.abs(var2_diff)
print abs_var2


####### Relative Humidity

# In[116]:

diff3 = np.zeros(data_len)

for x in range(data_len):
    diff3[x] = var3[x] - var3[x-1]
    
var3_diff = diff3[1:]
abs_var3 = np.abs(var3_diff)
print abs_var3


#### Deciding Events

# In[13]:

jump_1 = np.where(abs_var1 >= 0)
jump_2 = np.where(abs_var2 >= 0)
#jump_3 = np.where(abs_var3 >=5)


# In[14]:

print jump_1
print jump_2
#print jump_3


# In[15]:

#print var1[jump_1]
print var2[jump_2]
#print var3[jump_3]


#### Finding Event Points and Scales

# In[16]:

print len(var1[jump_1])
print len(var2[jump_2])
#print len(var3[jump_3])


#### Advancing the Index for Moment of Actual Jump

####### Temperature 

# In[17]:

new1 = [x+1 for x in jump_1]


# In[18]:

print new1
print jump_1


# In[19]:

print var1[new1]


# In[18]:

scale_1 = var1.max() - var1.min()


# In[19]:

print scale_1


# In[20]:

jump1_scale = var1[new1].max() - var1[new1].min()


####### Dewpoint

# In[21]:

new2 = [x+1 for x in jump_2]


# In[22]:

print var2[new2]


# In[23]:

jump2_scale = var2[new2].max() - var2[new2].min()


####### RH

# In[24]:

new3 = [x+1 for x in jump_3]


# In[25]:

jump3_scale = var3[new3].max() - var3[new3].min()


# In[26]:

print jump1_scale
print jump2_scale
print jump3_scale


### This is our music

# In[27]:

print var1[new1]
print var2[new2]
print var3[new3]


# In[24]:

var_1 = var1[new1]
var_2 = var2[new2]
#var_3 = var3[new3]


## Setting Middle C 

#### Finding Unique Values

# In[25]:

len1 = len(np.unique(var1[new1]))


# In[26]:

len2 = len(np.unique(var2[new2]))


# In[27]:

#len3 = len(np.unique(var3[new3]))


# In[28]:

np.unique(var1[new1])


# In[29]:

np.unique(var2[new2])


# In[34]:

np.unique(var3[new3])


# In[30]:

unq1 = np.unique(var1[new1])
unq2 = np.unique(var2[new2])
#unq3 = np.unique(var3[new3])


#### Middle C Points

# In[31]:

middlec_1 = np.around((jump1_scale/2.) + var1[new1].min())
middlec_2 = np.around((jump2_scale/2.) + var2[new2].min())
#middlec_3 = np.around((jump3_scale/2.) + var3[new3].min())


# In[32]:

print middlec_1
print middlec_2
#print middlec_3


#### Now we can subtract every value from middle C to see how far away on the musical scale:

####### Temp

# In[33]:

notes_1 = np.zeros(len1)

for x in range(len1):
    notes_1[x] = middlec_1 - np.unique(var1[new1])[x]
    
#notes_temp = np.abs(notes_t)


# In[34]:

print notes_1


# In[35]:

print notes_1
print np.unique(var1[new1])


####### Dew

# In[35]:

print len(np.unique(var2[new2]))


# In[36]:

notes_2 = np.zeros(len2)

for x in range(len2):
    notes_2[x] = middlec_2 - np.unique(var2[new2])[x]
    
#notes_dew = np.abs(notes_d)


# In[37]:

print notes_2
print np.unique(var2[new2])


###### RH

# In[44]:

print len(np.unique(var3[new3]))


# In[110]:

notes_3 = np.zeros(len3)

for x in range(len3):
    notes_3[x] = middlec_3 - np.unique(var3[new3])[x]
    
#notes_relh = np.abs(notes_rh)


# In[111]:

print notes_3
print np.unique(var3[new3])


#### Respective Distance from Middle C

# In[38]:

print notes_1
print notes_2
#print notes_3


# In[39]:

print middlec_1
print middlec_2
#print middlec_3


####### Keep in mind now that these are only for the unique points, so there are some repeats in the full set

#### These are the indices for the music in the whole dataset: 

# In[40]:

print new1
print new2
#print new3


#### Putting it all together:

# In[50]:

print notes_1
print notes_2
print notes_3


####### Temperature 

# In[44]:

print notes_1
print np.unique(var1[new1])
print var1[new1]


# In[41]:

dist1 = np.array([unq1,notes_1])


# In[42]:

#print dist1
#print dist1[0,:]


# In[43]:

var1len = len(var_1)
print var1len


# In[44]:

fill1 = np.arange(var1len)


# In[45]:

dist_1 = [dist1[1,np.where(dist1[0,:]==var_1[x])] for x in fill1]


# In[46]:

var1_notes = np.zeros(var1len)
for x in range(var1len):
    var1_notes[x] = dist_1[x]
    #print dist_t[x]


# In[47]:

print var1_notes


# In[68]:

print var1_notes
print new1


# In[48]:

print middlec_1
print middlec_2
#print middlec_3


####### Dewpoint

# In[49]:

dist2 = np.array([unq2,notes_2])


# In[50]:

var2len = len(var_2)
print var2len


# In[51]:

fill2 = np.arange(var2len)
dist_2 = [dist2[1,np.where(dist2[0,:]==var_2[x])] for x in fill2]


# In[52]:

var2_notes = np.zeros(var2len)
for x in range(var2len):
    var2_notes[x] = dist_2[x]


# In[53]:

print var2_notes


# In[54]:

print var2_notes
print new2


####### RH

# In[67]:

dist3 = np.array([unq3,notes_3])
var3len = len(var_3)
print var3len


# In[68]:

fill3 = np.arange(var3len)
dist_3 = [dist3[1,np.where(dist3[0,:]==var_3[x])] for x in fill3]


# In[69]:

var3_notes = np.zeros(var3len)
for x in range(var3len):
    var3_notes[x] = dist_3[x]


# In[70]:

print var3_notes


# In[71]:

print var3_notes
print new3


# In[72]:

print len(var3_notes)
print len(var3[jump_3])
print len(var3[new3]) 


# In[73]:

print type(var3_notes)


#### Transferring to Music Scale 

# In[55]:

events_1 = np.ravel(new1)
events_2 = np.ravel(new2)
#events_3 = np.ravel(new3)


# In[56]:

mat_1 = np.array([var1_notes,events_1])
mat_2 = np.array([var2_notes,events_2])
#mat_3 = np.array([var3_notes,events_3])


# In[57]:

print len(events_2)


# In[58]:

print mat_2[:,0]


# In[59]:

for x in range(241):
    print mat_2[:,x]  


# In[50]:

print var1_notes


#### Trying to print for lilypad

# In[60]:

auto = np.arange(40)
print auto


# In[61]:

a = auto[5::7]
b = auto[6::7]
c = auto[0::7]
d = auto[1::7]
e = auto[2::7]
f = auto[3::7]
g = auto[4::7]


# In[62]:

neg = np.arange(40)* -1
autoneg = neg[1:] 
print autoneg


# In[63]:

aneg = autoneg[1::7]
bneg = autoneg[0::7]
cneg = autoneg[6::7]
dneg = autoneg[5::7]
eneg = autoneg[4::7]
fneg = autoneg[3::7]
gneg = autoneg[2::7]


# In[64]:

scale_a = np.append(aneg,a)
scale_b = np.append(bneg,b)
scale_c = np.append(cneg,c)
scale_d = np.append(dneg,d)
scale_e = np.append(eneg,e)
scale_f = np.append(fneg,f)
scale_g = np.append(gneg,g)


# In[65]:

len_a = len(scale_a)
len_b = len(scale_b)
len_c = len(scale_c)
len_d = len(scale_d)
len_e = len(scale_e)
len_f = len(scale_f)
len_g = len(scale_g)


# In[66]:

loop_a = np.arange(len_a)
loop_b = np.arange(len_b)
loop_c = np.arange(len_c)
loop_d = np.arange(len_d)
loop_e = np.arange(len_e)
loop_f = np.arange(len_f)
loop_g = np.arange(len_g)


####### Temperature

# In[67]:

lily1 = np.rint(var1_notes)*-1
#lilyrr = var1_notes*-1


# In[68]:

print lily1


# In[69]:

mask_a1 = np.in1d(lily1, scale_a)
mask_b1 = np.in1d(lily1, scale_b)
mask_c1 = np.in1d(lily1, scale_c)
mask_d1 = np.in1d(lily1, scale_d)
mask_e1 = np.in1d(lily1, scale_e)
mask_f1 = np.in1d(lily1, scale_f)
mask_g1 = np.in1d(lily1, scale_g)


# In[70]:

sel_a1 = lily1[mask_a1]
sel_b1 = lily1[mask_b1]
sel_c1 = lily1[mask_c1]
sel_d1 = lily1[mask_d1]
sel_e1 = lily1[mask_e1]
sel_f1 = lily1[mask_f1]
sel_g1 = lily1[mask_g1]


# In[71]:

lily1[mask_a1] = 99
lily1[mask_b1] = 98
lily1[mask_c1] = 97
lily1[mask_d1] = 96
lily1[mask_e1] = 95
lily1[mask_f1] = 94
lily1[mask_g1] = 93
print lily1


# In[72]:

cond1 = [lily1==99,lily1==98,lily1==97,lily1==96,lily1==95,lily1==94,lily1==93]
choice1 = ["a4","b4","c4","d4","e4","f4","g4"]
final1 = np.select(cond1, choice1)
print final1


# In[65]:

final1s = np.array2string(final1)


#### Lilypad Dew

# In[82]:

#lily2 = var2_notes*-1
lily2 = np.rint(var2_notes)*-1
#print var2_notes
print lily2


# In[74]:

mask_a2 = np.in1d(lily2, scale_a)
mask_b2 = np.in1d(lily2, scale_b)
mask_c2 = np.in1d(lily2, scale_c)
mask_d2 = np.in1d(lily2, scale_d)
mask_e2 = np.in1d(lily2, scale_e)
mask_f2 = np.in1d(lily2, scale_f)
mask_g2 = np.in1d(lily2, scale_g)


# In[75]:

sel_a2 = lily2[mask_a2]
sel_b2 = lily2[mask_b2]
sel_c2 = lily2[mask_c2]
sel_d2 = lily2[mask_d2]
sel_e2 = lily2[mask_e2]
sel_f2 = lily2[mask_f2]
sel_g2 = lily2[mask_g2]


# In[76]:

lily2[mask_a2] = 99
lily2[mask_b2] = 98
lily2[mask_c2] = 97
lily2[mask_d2] = 96
lily2[mask_e2] = 95
lily2[mask_f2] = 94
lily2[mask_g2] = 93
print lily2


# In[77]:

cond2 = [lily2==99,lily2==98,lily2==97,lily2==96,lily2==95,lily2==94,lily2==93]
choice2 = ["a4","b4","c4","d4","e4","f4","g4"]
final2 = np.select(cond2, choice2)
print final2


#### Lilypad RH

# In[98]:

lily3 = var3_notes*-1


# In[99]:

mask_a3 = np.in1d(lily3, scale_a)
mask_b3 = np.in1d(lily3, scale_b)
mask_c3 = np.in1d(lily3, scale_c)
mask_d3 = np.in1d(lily3, scale_d)
mask_e3 = np.in1d(lily3, scale_e)
mask_f3 = np.in1d(lily3, scale_f)
mask_g3 = np.in1d(lily3, scale_g)


# In[100]:

sel_a3 = lily3[mask_a3]
sel_b3 = lily3[mask_b3]
sel_c3 = lily3[mask_c3]
sel_d3 = lily3[mask_d3]
sel_e3 = lily3[mask_e3]
sel_f3 = lily3[mask_f3]
sel_g3 = lily3[mask_g3]


# In[101]:

lily3[mask_a3] = 99
lily3[mask_b3] = 98
lily3[mask_c3] = 97
lily3[mask_d3] = 96
lily3[mask_e3] = 95
lily3[mask_f3] = 94
lily3[mask_g3] = 93
print lily3


# In[102]:

cond3 = [lily3==99,lily3==98,lily3==97,lily3==96,lily3==95,lily3==94,lily3==93]
choice3 = ["a4","b4","c4","d4","e4","f4","g4"]
final3 = np.select(cond3, choice3)
print final3


# In[102]:




# In[102]:




# In[102]:




# In[ ]:



