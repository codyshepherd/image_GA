import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import json
import os
from PIL import Image, ImageDraw

names = ['steps','pop','poly','shape','child','hard','med','soft','delt','fitness']

files = [f for f in os.listdir('.') if os.path.isfile(f)]

#for f in files:
#    print(str(f))

txts = [f for f in files if f[-4:]=='.txt']
#pics = [f for f in files if f[-4:]=='.png']

for t in txts:
    if t == 'requirements.txt':
        continue
    print(str(t))
    with open(str(t), 'r') as fh:
        stuff = json.load(fh)
    tracker = [float(f) for f in stuff['tracker'].strip('[]').split(', ')]
    if len(tracker) > 100:
        tracker = tracker[:-1]
    steps = int(stuff['steps'])
    interval = steps//100
    figname = ''
    for n in names:
        figname += n
        figname += stuff[n]
    plt.figure()
    plt.plot(range(0, steps, interval), tracker)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.grid()
    plt.title(figname)
    plt.savefig(figname + '.png')
    plt.close

    handle = t[:-4]
    pic = handle + '.png'
    os.rename(pic, figname + '_img.png')
