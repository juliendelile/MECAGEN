import matplotlib.pyplot as plt
import numpy as np
import sys

plt.rc('text', usetex=False)
plt.rc('font', family='Arial')

##############################
##### Read command line ######
##############################

#python grnstate_plot_v2.py --tmin 1 --tmax 99 --folder_path /data/Work/Publications/2014_Methodo_MecaGen/Mecagen\ Methodo\ paper\ \(J.\ Delile\)/schematics/Figure3/data_smts/ --id 300 --color  /data/Work/Publications/2014_Methodo_MecaGen/Mecagen\ Methodo\ paper\ \(J.\ Delile\)/schematics/Figure3/color.csv --filter_path /data/Work/Publications/2014_Methodo_MecaGen/Mecagen\ Methodo\ paper\ \(J.\ Delile\)/schematics/Figure3/filter.csv --output_path /home/julien/Desktop/test.png

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

ts_start = options.tmin         
ts_end   = options.tmax+1       
path     = options.folder_path  
idcell   = options.id           

##############################
######    Load data     ######
##############################

data_temp = np.genfromtxt(path+'/grnstate_t%04d' % (1)+'.csv', dtype=None, delimiter=';', names=True, deletechars='') 
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

# plt.figure(figsize=(8, 6), dpi = 300)
# plt.figure(figsize=(1.8, 1.8), dpi = 300)
plt.figure(figsize=(8.15, 2.15), dpi = 300)

plt.grid(linewidth=.3)

import itertools
marker = itertools.cycle(('+', '.', '*')) 

import random
for i, name in enumerate(molnames):
  # c=[random.random(),random.random(),random.random()]
  # print color_data[i]
  if filter_data[i]==[1.0]:
    # plt.plot(ts_space, molq[name][ts_space], marker = marker.next(), color=color_data[i], linestyle='-', markersize=6, markeredgecolor=None, markeredgewidth=0, linewidth=1)
    plt.plot(ts_space, molq[name][ts_space],color=color_data[i], linestyle='-', linewidth=1.0)
  
ymax = 0
for idx, name in enumerate(molnames):
  if filter_data[idx]==[1.0]:
    ymax = max(max(molq[name]), ymax)

plt.ylim(-.05 * ymax, 1.05 * ymax)

# Shink current axis by 20%
ax = plt.subplot(111)
box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.set_position([box.x0 + .2 * box.width , box.y0, box.width * 0.8, box.height])
ax.set_position([box.x0, box.y0, box.width, box.height])

molnames_dollar=[0]*len(molnames) #{}
i=0
for idx, name in enumerate(molnames):
    if filter_data[idx]==[1.0]:
      # molnames_dollar[i] = '$'+name+'$'
      molnames_dollar[i] = name
      i=i+1

# Put a legend to the right of the current axis
# ax.set_position([box.x0 , box.y0, box.width * 0.2, box.height])
# ax.legend(molnames_dollar, loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 8, ncol=4)

# Labels
# plt.xlabel('Time (min)', fontsize = 8)
# plt.ylabel('Concentration (A.U.)', fontsize = 8)

plt.tick_params(axis='both', which='major', labelsize=8)
plt.tick_params(axis='both', which='minor', labelsize=8)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_linewidth(0.3)
ax.spines['left'].set_linewidth(0.3)

plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    right="off",
    left="off"
    ) # labels along the bottom edge are off

plt.savefig(options.output_path, dpi=300)

plt.show()