import matplotlib.pyplot as plt
import numpy as np
import sys

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

##############################
##### Read command line ######
##############################

# python grnstate_plot.py --tmin 1 --tmax 99 --folder_path /home/julien/Desktop/data_smts/ --id 300 --color_path /home/julien/Desktop/color.csv

import optparse

parser = optparse.OptionParser()
parser.add_option('--tmin',
                  action="store",
                  dest="tmin", 
                  type="int"
                  )
parser.add_option('--tmax',
                  action="store",
                  dest="tmax",
                  type="int"
                  )
parser.add_option('--folder_path',
                  action="store",
                  dest="folder_path"
                  )
parser.add_option('--id',
                  action="store",
                  dest="id",
                  type="int"
                  )
parser.add_option('--color_path',
                  action="store",
                  dest="color_path"
                  )
parser.add_option('--filter_path',
                  action="store",
                  dest="filter_path"
                  )
parser.add_option('-o', '--output_path',
                  action="store",
                  dest="output_path"
                  )
options, remainder = parser.parse_args()

ts_start = options.tmin         #int(sys.argv[1])
ts_end   = options.tmax+1       #int(sys.argv[2])
path     = options.folder_path  #sys.argv[3]
idcell   = options.id           #int(sys.argv[4])

# print ts_end
# print ts_start
# print path
# print idcell
##############################
######    Load data     ######
##############################

# data_temp = np.genfromtxt(path+'/grnstate_t%04d' % (1)+'.csv', dtype=float, delimiter=';', names=True) 
data_temp = np.genfromtxt(path+'/grnstate_t%04d' % (1)+'.csv', dtype=None, delimiter=';', names=True, deletechars='') 
# print data_temp
molnames_temp = data_temp.dtype.names
molnames = molnames_temp[1:len(molnames_temp)]
print molnames

molq = {}
for n in molnames:
  molq[n] = np.zeros((ts_end-ts_start+1))

ts_space = np.arange(ts_start,ts_end)
for ts in ts_space:
  data = np.genfromtxt(path+'/grnstate_t%04d' % (ts)+'.csv', dtype=None, delimiter=';', names=True, deletechars='') 

  for n in molnames:
    molq[n][ts] = data[idcell][n];

##############################
######    Load colors    ######
##############################

if options.color_path:
    import csv
    with open(options.color_path, 'Ur') as f:
      color_data = list(tuple(t) for t in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC, delimiter=','))
    for i, col in enumerate(color_data):
      color_data[i] = [col[0]/255.0, col[1]/255.0, col[2]/255.0 ]

else:
    import random
    color_data={}
    for idx, name in enumerate(molnames):
      color_data[idx]=[random.random(),random.random(),random.random()]


##############################
######    Filter       # #####
##############################

if options.filter_path:
    import csv
    with open(options.filter_path, 'Ur') as f:
      filter_data = list(t for t in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC))

else:
    filter_data={}
    for idx, name in enumerate(molnames):
      filter_data[idx]=[1.0]

##############################
#####         Plot     #######
##############################

plt.figure(figsize=(8, 6), dpi = 300)

plt.grid()

import itertools
marker = itertools.cycle(('+', '.', '*')) 

import random
for i, name in enumerate(molnames):
  # c=[random.random(),random.random(),random.random()]
  # print color_data[i]
  if filter_data[i]==[1.0]:
    plt.plot(ts_space, molq[name][ts_space], marker = marker.next(), color=color_data[i], linestyle='-', markersize=6, markeredgecolor=None, markeredgewidth=0, linewidth=1)
  
ymax = 0
for idx, name in enumerate(molnames):
  if filter_data[idx]==[1.0]:
    ymax = max(max(molq[name]), ymax)

plt.ylim(-.05 * ymax, 1.05 * ymax)
# plt.legend(molnames, loc='lower right', bbox_to_anchor = (.85,.1) )
# plt.legend(molnames, loc='lower right')

# Shink current axis by 20%
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

molnames_dollar=[0]*len(molnames) #{}
i=0
for idx, name in enumerate(molnames):
    if filter_data[idx]==[1.0]:
      # molnames_dollar[i] = '$'+name+'$'
      molnames_dollar[i] = name
      i=i+1

# print type(molnames_dollar)
# print molnames_dollar
# print type(molnames)
# print molnames
# Put a legend to the right of the current axis
ax.legend(molnames_dollar, loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 12)

plt.xlabel('Time (min)', fontsize = 12)
plt.ylabel('Concentration (m^{-3})', fontsize = 12)


# figure = plt.gcf()
# plt.figure(num=None, figsize=(2, 1), dpi=300, facecolor='w', edgecolor='k')

# plt.savefig("/home/julien/Desktop/stms_curves.pdf")
# plt.savefig("/home/julien/Desktop/stms_curves.png", dpi=300)

# plt.savefig(options.output_path, dpi=300)

plt.show()