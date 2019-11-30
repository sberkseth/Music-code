"""
    
=====================================
Upload Data Files for Sonification
=====================================
    
Import files to create output used for data sonification. Output is a series of music note identifiers, which then need to be uploaded to Lilypad program for sonification.     
"""

# import packages

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.optimize import curve_fit
get_ipython().magic(u'matplotlib inline')

#### Import Data File

from pandas import read_csv
data_load = read_csv('barry_recon2.csv')

data=np.loadtxt('barry_recon2.csv',delimiter=',',skiprows=1)

#listed below are examples of variations that can be used depending on data file
#data = np.loadtxt('barry_data1.csv',delimiter=',',skiprows=0) 
#data = np.loadtxt('barry_recon.dat') 

print data[:,9]

var1 = data[:,5]
coffee = data[:,9]
#var3 = data[:,2]

data_len = len(var1)

var2 = coffee/2.
print var2
#print coffee

print var2.min()

#plt.plot(var1, 'r--', var2, 'g--', var3, 'b--')
plt.plot(var1, 'r--', var2, 'g--')

#Used for creating a quick animation (until I can figure out ffmpeg..)

#import matplotlib.animation as animation
#fig, ax = plt.subplots()

#t= np.arange(0,242,1)
#ax.set_xlabel('time')
#ax.set_ylabel('Pressure (mb) ', color='r')
#ax.plot(t,var1,'r')
#ax.tick_params(axis='y', colors='r')
#ax.axes.get_xaxis().set_visible(False)

#ax2 = ax.twinx()
#ax2.set_ylabel('Peak 10s Wind (kt)', color='b')  # we already handled the x-label with ax1
#ax2.plot(t,var2, 'b')
#ax2.tick_params(axis='y', colors='b')
#plt.axvline(    x=242,color='g')
#plt.savefig("barry242.png")


### 

#### Finding the Variances from Point to Point (Changes over Time)

####### Temperature

diff1 = np.zeros(data_len)

for x in range(data_len):
    diff1[x] = var1[x] - var1[x-1]

var1_diff = diff1[1:]
abs_var1 = np.abs(var1_diff)
print abs_var1


####### Dewpoint

diff2 = np.zeros(data_len)

for x in range(data_len):
    diff2[x] = var2[x] - var2[x-1]
    
var2_diff = diff2[1:]
abs_var2 = np.abs(var2_diff)
print abs_var2


####### Relative Humidity

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

new1 = [x+1 for x in jump_1]

print new1
print jump_1

print var1[new1]

scale_1 = var1.max() - var1.min()

print scale_1

jump1_scale = var1[new1].max() - var1[new1].min()


####### Dewpoint

new2 = [x+1 for x in jump_2]

print var2[new2]

jump2_scale = var2[new2].max() - var2[new2].min()


####### RH


new3 = [x+1 for x in jump_3]


jump3_scale = var3[new3].max() - var3[new3].min()


print jump1_scale
print jump2_scale
print jump3_scale


### This is our music

print var1[new1]
print var2[new2]
print var3[new3]

var_1 = var1[new1]
var_2 = var2[new2]
#var_3 = var3[new3]


## Setting Middle C 

#### Finding Unique Values


len1 = len(np.unique(var1[new1]))

len2 = len(np.unique(var2[new2]))

#len3 = len(np.unique(var3[new3]))

np.unique(var1[new1])

np.unique(var2[new2])

np.unique(var3[new3])

unq1 = np.unique(var1[new1])
unq2 = np.unique(var2[new2])
#unq3 = np.unique(var3[new3])


#### Middle C Points

middlec_1 = np.around((jump1_scale/2.) + var1[new1].min())
middlec_2 = np.around((jump2_scale/2.) + var2[new2].min())
#middlec_3 = np.around((jump3_scale/2.) + var3[new3].min())


print middlec_1
print middlec_2
#print middlec_3


#### Now we can subtract every value from middle C to see how far away on the musical scale:

####### Temp

notes_1 = np.zeros(len1)

for x in range(len1):
    notes_1[x] = middlec_1 - np.unique(var1[new1])[x]
    
#notes_temp = np.abs(notes_t)

print notes_1

print notes_1
print np.unique(var1[new1])


####### Dew

print len(np.unique(var2[new2]))

notes_2 = np.zeros(len2)

for x in range(len2):
    notes_2[x] = middlec_2 - np.unique(var2[new2])[x]
    
#notes_dew = np.abs(notes_d)

print notes_2
print np.unique(var2[new2])


###### RH

print len(np.unique(var3[new3]))

notes_3 = np.zeros(len3)

for x in range(len3):
    notes_3[x] = middlec_3 - np.unique(var3[new3])[x]
    
#notes_relh = np.abs(notes_rh)


print notes_3
print np.unique(var3[new3])


#### Respective Distance from Middle C


print notes_1
print notes_2
#print notes_3

print middlec_1
print middlec_2
#print middlec_3


####### Keep in mind now that these are only for the unique points, so there are some repeats in the full set

#### These are the indices for the music in the whole dataset: 


print new1
print new2
#print new3


#### Putting it all together:

print notes_1
print notes_2
print notes_3


####### Temperature 

print notes_1
print np.unique(var1[new1])
print var1[new1]

dist1 = np.array([unq1,notes_1])

#print dist1
#print dist1[0,:]


var1len = len(var_1)
print var1len


fill1 = np.arange(var1len)

dist_1 = [dist1[1,np.where(dist1[0,:]==var_1[x])] for x in fill1]


var1_notes = np.zeros(var1len)
for x in range(var1len):
    var1_notes[x] = dist_1[x]
    #print dist_t[x]


print var1_notes


print var1_notes
print new1

print middlec_1
print middlec_2
#print middlec_3


####### Dewpoint

dist2 = np.array([unq2,notes_2])

var2len = len(var_2)
print var2len


fill2 = np.arange(var2len)
dist_2 = [dist2[1,np.where(dist2[0,:]==var_2[x])] for x in fill2]


var2_notes = np.zeros(var2len)
for x in range(var2len):
    var2_notes[x] = dist_2[x]


print var2_notes


print var2_notes
print new2


####### RH

dist3 = np.array([unq3,notes_3])
var3len = len(var_3)
print var3len

fill3 = np.arange(var3len)
dist_3 = [dist3[1,np.where(dist3[0,:]==var_3[x])] for x in fill3]


var3_notes = np.zeros(var3len)
for x in range(var3len):
    var3_notes[x] = dist_3[x]

print var3_notes


print var3_notes
print new3

print len(var3_notes)
print len(var3[jump_3])
print len(var3[new3]) 

print type(var3_notes)


#### Transferring to Music Scale 


events_1 = np.ravel(new1)
events_2 = np.ravel(new2)
#events_3 = np.ravel(new3)


mat_1 = np.array([var1_notes,events_1])
mat_2 = np.array([var2_notes,events_2])
#mat_3 = np.array([var3_notes,events_3])


print len(events_2)


print mat_2[:,0]

for x in range(241):
    print mat_2[:,x]  

print var1_notes


#### Trying to print for lilypad

auto = np.arange(40)
print auto

a = auto[5::7]
b = auto[6::7]
c = auto[0::7]
d = auto[1::7]
e = auto[2::7]
f = auto[3::7]
g = auto[4::7]


neg = np.arange(40)* -1
autoneg = neg[1:] 
print autoneg

aneg = autoneg[1::7]
bneg = autoneg[0::7]
cneg = autoneg[6::7]
dneg = autoneg[5::7]
eneg = autoneg[4::7]
fneg = autoneg[3::7]
gneg = autoneg[2::7]


scale_a = np.append(aneg,a)
scale_b = np.append(bneg,b)
scale_c = np.append(cneg,c)
scale_d = np.append(dneg,d)
scale_e = np.append(eneg,e)
scale_f = np.append(fneg,f)
scale_g = np.append(gneg,g)


len_a = len(scale_a)
len_b = len(scale_b)
len_c = len(scale_c)
len_d = len(scale_d)
len_e = len(scale_e)
len_f = len(scale_f)
len_g = len(scale_g)


loop_a = np.arange(len_a)
loop_b = np.arange(len_b)
loop_c = np.arange(len_c)
loop_d = np.arange(len_d)
loop_e = np.arange(len_e)
loop_f = np.arange(len_f)
loop_g = np.arange(len_g)


####### Temperature


lily1 = np.rint(var1_notes)*-1
#lilyrr = var1_notes*-1


print lily1

mask_a1 = np.in1d(lily1, scale_a)
mask_b1 = np.in1d(lily1, scale_b)
mask_c1 = np.in1d(lily1, scale_c)
mask_d1 = np.in1d(lily1, scale_d)
mask_e1 = np.in1d(lily1, scale_e)
mask_f1 = np.in1d(lily1, scale_f)
mask_g1 = np.in1d(lily1, scale_g)


sel_a1 = lily1[mask_a1]
sel_b1 = lily1[mask_b1]
sel_c1 = lily1[mask_c1]
sel_d1 = lily1[mask_d1]
sel_e1 = lily1[mask_e1]
sel_f1 = lily1[mask_f1]
sel_g1 = lily1[mask_g1]


lily1[mask_a1] = 99
lily1[mask_b1] = 98
lily1[mask_c1] = 97
lily1[mask_d1] = 96
lily1[mask_e1] = 95
lily1[mask_f1] = 94
lily1[mask_g1] = 93
print lily1


cond1 = [lily1==99,lily1==98,lily1==97,lily1==96,lily1==95,lily1==94,lily1==93]
choice1 = ["a4","b4","c4","d4","e4","f4","g4"]
final1 = np.select(cond1, choice1)
print final1


final1s = np.array2string(final1)


#### Lilypad Dew

#lily2 = var2_notes*-1
lily2 = np.rint(var2_notes)*-1
#print var2_notes
print lily2

mask_a2 = np.in1d(lily2, scale_a)
mask_b2 = np.in1d(lily2, scale_b)
mask_c2 = np.in1d(lily2, scale_c)
mask_d2 = np.in1d(lily2, scale_d)
mask_e2 = np.in1d(lily2, scale_e)
mask_f2 = np.in1d(lily2, scale_f)
mask_g2 = np.in1d(lily2, scale_g)

sel_a2 = lily2[mask_a2]
sel_b2 = lily2[mask_b2]
sel_c2 = lily2[mask_c2]
sel_d2 = lily2[mask_d2]
sel_e2 = lily2[mask_e2]
sel_f2 = lily2[mask_f2]
sel_g2 = lily2[mask_g2]


lily2[mask_a2] = 99
lily2[mask_b2] = 98
lily2[mask_c2] = 97
lily2[mask_d2] = 96
lily2[mask_e2] = 95
lily2[mask_f2] = 94
lily2[mask_g2] = 93
print lily2

cond2 = [lily2==99,lily2==98,lily2==97,lily2==96,lily2==95,lily2==94,lily2==93]
choice2 = ["a4","b4","c4","d4","e4","f4","g4"]
final2 = np.select(cond2, choice2)
print final2


#### Lilypad RH

lily3 = var3_notes*-1


mask_a3 = np.in1d(lily3, scale_a)
mask_b3 = np.in1d(lily3, scale_b)
mask_c3 = np.in1d(lily3, scale_c)
mask_d3 = np.in1d(lily3, scale_d)
mask_e3 = np.in1d(lily3, scale_e)
mask_f3 = np.in1d(lily3, scale_f)
mask_g3 = np.in1d(lily3, scale_g)


sel_a3 = lily3[mask_a3]
sel_b3 = lily3[mask_b3]
sel_c3 = lily3[mask_c3]
sel_d3 = lily3[mask_d3]
sel_e3 = lily3[mask_e3]
sel_f3 = lily3[mask_f3]
sel_g3 = lily3[mask_g3]


lily3[mask_a3] = 99
lily3[mask_b3] = 98
lily3[mask_c3] = 97
lily3[mask_d3] = 96
lily3[mask_e3] = 95
lily3[mask_f3] = 94
lily3[mask_g3] = 93
print lily3

cond3 = [lily3==99,lily3==98,lily3==97,lily3==96,lily3==95,lily3==94,lily3==93]
choice3 = ["a4","b4","c4","d4","e4","f4","g4"]
final3 = np.select(cond3, choice3)
print final3


