# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 13:42:45 2021

@author: Robin
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import dill
import fnmatch



filelist = fnmatch.filter(os.listdir('samplers/'), '*_sampler.pickle')
filelist_gauss = fnmatch.filter(os.listdir('samplers_astrosil/'), '*_sampler.pickle')
filelist_10ice = fnmatch.filter(os.listdir('samplers_10ice/'), '*_sampler.pickle')
filelist_5ice = fnmatch.filter(os.listdir('samplers_5ice/'), '*_sampler.pickle')
filelist_20ice = fnmatch.filter(os.listdir('samplers_20ice/'), '*_sampler.pickle')



best_guesses = [[], [], [], []]
best_guesses_gauss = [[], [], [], [], [], [], []]
best_guesses_10ice = [[], [], [], [], [], [], []]
best_guesses_20ice = [[], [], [], [], [], [], []]
best_guesses_5ice = [[], [], [], [], [], [], []]

best_guesses_gauss_qfix = [[], [], [], [], [], [], []]
best_guesses_10ice_qfix = [[], [], [], [], [], [], []]
best_guesses_20ice_qfix = [[], [], [], [], [], [], []]
best_guesses_5ice_qfix = [[], [], [], [], [], [], []]

weird_q = []

for x in filelist:
    
    jsonname = x.split('_')[0]
    jsonfile = open("processed_data/{:}".format(jsonname + '.json'))
    inputjson = json.load(jsonfile)
    
    picklefile = open('samplers/{:}'.format(x), 'rb')
    sampler = dill.load(picklefile)
    samples = sampler.flatchain
    theta_max = samples[np.argmax(sampler.flatlnprobability)]
    
    if len(theta_max) == 3:
        if theta_max[1] >= -3:
            weird_q.append(x)
        if theta_max[1] <= -4:
            weird_q.append(x)
    
    if len(theta_max) == 3:
        best_guesses[0].append(theta_max[0])
        best_guesses[1].append(theta_max[1])
        best_guesses[2].append(theta_max[2])
        best_guesses[3].append(inputjson['main_results'][0]['lstar'])
    
    if len(theta_max) == 2:
        best_guesses[0].append(theta_max[0])
        best_guesses[1].append(-3.5)
        best_guesses[2].append(theta_max[1])
        best_guesses[3].append(inputjson['main_results'][0]['lstar'])
        
        
aaa = [[x[0] for x in best_guesses_gauss[4]], [x[1] for x in best_guesses_gauss[4]]]

for x in filelist_gauss:
    
    jsonname = x.split('_')[0]
    jsonfile = open("processed_data/{:}".format(jsonname + '.json'))
    inputjson = json.load(jsonfile)
    
    picklefile = open('samplers_astrosil/{:}'.format(x), 'rb')
    sampler = dill.load(picklefile)
    samples = sampler.flatchain
    theta_max = samples[np.argmax(sampler.flatlnprobability)]
    flat_samples = sampler.get_chain(discard=100, flat=True)

    if len(theta_max) == 3:
        best_guesses_gauss[0].append(theta_max[0])
        best_guesses_gauss[1].append(theta_max[1])
        best_guesses_gauss[2].append(theta_max[2])
        best_guesses_gauss[3].append(inputjson['main_results'][0]['lstar'])
        best_guesses_gauss[4].append(np.percentile(flat_samples[:, 0], [16, 84]))
        best_guesses_gauss[5].append(np.percentile(flat_samples[:, 1], [16, 84]))
        best_guesses_gauss[6].append(np.percentile(flat_samples[:, 2], [16, 84]))

    if len(theta_max) == 2:
        # best_guesses_gauss[0].append(theta_max[0])
        # best_guesses_gauss[1].append(-3.5)
        # best_guesses_gauss[2].append(theta_max[1])
        # best_guesses_gauss[3].append(inputjson['main_results'][0]['lstar'])
        
        best_guesses_gauss_qfix[0].append(theta_max[0])
        best_guesses_gauss_qfix[1].append(-3.5)
        best_guesses_gauss_qfix[2].append(theta_max[1])
        best_guesses_gauss_qfix[3].append(inputjson['main_results'][0]['lstar'])
        best_guesses_gauss_qfix[4].append(np.percentile(flat_samples[:, 0], [16, 84]))
        best_guesses_gauss_qfix[5].append([0, 0])
        best_guesses_gauss_qfix[6].append(np.percentile(flat_samples[:, 1], [16, 84]))
        
best_guesses_gauss[4] = [[x[0] for x in best_guesses_gauss[4]], [x[1] for x in best_guesses_gauss[4]]]
best_guesses_gauss[5] = [[x[0] for x in best_guesses_gauss[5]], [x[1] for x in best_guesses_gauss[5]]]
best_guesses_gauss[6] = [[x[0] for x in best_guesses_gauss[6]], [x[1] for x in best_guesses_gauss[6]]]
best_guesses_gauss_qfix[4] = [[x[0] for x in best_guesses_gauss_qfix[4]], [x[1] for x in best_guesses_gauss_qfix[4]]]
best_guesses_gauss_qfix[5] = [[x[0] for x in best_guesses_gauss_qfix[5]], [x[1] for x in best_guesses_gauss_qfix[5]]]
best_guesses_gauss_qfix[6] = [[x[0] for x in best_guesses_gauss_qfix[6]], [x[1] for x in best_guesses_gauss_qfix[6]]]

for x in filelist_10ice:
    
    jsonname = x.split('_')[0]
    jsonfile = open("processed_data/{:}".format(jsonname + '.json'))
    inputjson = json.load(jsonfile)
    
    picklefile = open('samplers_10ice/{:}'.format(x), 'rb')
    sampler = dill.load(picklefile)
    samples = sampler.flatchain
    theta_max = samples[np.argmax(sampler.flatlnprobability)]
    flat_samples = sampler.get_chain(discard=100, flat=True)


    if len(theta_max) == 3:
        best_guesses_10ice[0].append(theta_max[0])
        best_guesses_10ice[1].append(theta_max[1])
        best_guesses_10ice[2].append(theta_max[2])
        best_guesses_10ice[4].append(np.percentile(flat_samples[:, 0], [16, 84]))
        best_guesses_10ice[5].append(np.percentile(flat_samples[:, 1], [16, 84]))
        best_guesses_10ice[6].append(np.percentile(flat_samples[:, 2], [16, 84]))

        best_guesses_10ice[3].append(inputjson['main_results'][0]['lstar'])

    if len(theta_max) == 2:
        # best_guesses_10ice[0].append(theta_max[0])
        # best_guesses_10ice[1].append(-3.5)
        # best_guesses_10ice[2].append(theta_max[1])
        # best_guesses_10ice[3].append(inputjson['main_results'][0]['lstar'])
        
        best_guesses_10ice_qfix[0].append(theta_max[0])
        best_guesses_10ice_qfix[1].append(-3.5)
        best_guesses_10ice_qfix[2].append(theta_max[1])
        best_guesses_10ice_qfix[4].append(np.percentile(flat_samples[:, 0], [16, 84]))
        best_guesses_10ice_qfix[5].append([0, 0])
        best_guesses_10ice_qfix[6].append(np.percentile(flat_samples[:, 1], [16, 84]))

        best_guesses_10ice_qfix[3].append(inputjson['main_results'][0]['lstar'])
        
best_guesses_10ice[4] = [[x[0] for x in best_guesses_10ice[4]], [x[1] for x in best_guesses_10ice[4]]]
best_guesses_10ice[5] = [[x[0] for x in best_guesses_10ice[5]], [x[1] for x in best_guesses_10ice[5]]]
best_guesses_10ice[6] = [[x[0] for x in best_guesses_10ice[6]], [x[1] for x in best_guesses_10ice[6]]]
best_guesses_10ice_qfix[4] = [[x[0] for x in best_guesses_10ice_qfix[4]], [x[1] for x in best_guesses_10ice_qfix[4]]]
best_guesses_10ice_qfix[5] = [[x[0] for x in best_guesses_10ice_qfix[5]], [x[1] for x in best_guesses_10ice_qfix[5]]]
best_guesses_10ice_qfix[6] = [[x[0] for x in best_guesses_10ice_qfix[6]], [x[1] for x in best_guesses_10ice_qfix[6]]]

for x in filelist_20ice:
    
    jsonname = x.split('_')[0]
    jsonfile = open("processed_data/{:}".format(jsonname + '.json'))
    inputjson = json.load(jsonfile)
    
    picklefile = open('samplers_20ice/{:}'.format(x), 'rb')
    sampler = dill.load(picklefile)
    samples = sampler.flatchain
    theta_max = samples[np.argmax(sampler.flatlnprobability)]
    flat_samples = sampler.get_chain(discard=100, flat=True)

    if len(theta_max) == 3:
        best_guesses_20ice[0].append(theta_max[0])
        best_guesses_20ice[1].append(theta_max[1])
        best_guesses_20ice[2].append(theta_max[2])
        best_guesses_20ice[4].append(np.percentile(flat_samples[:, 0], [16, 84]))
        best_guesses_20ice[5].append(np.percentile(flat_samples[:, 1], [16, 84]))
        best_guesses_20ice[6].append(np.percentile(flat_samples[:, 2], [16, 84]))

        best_guesses_20ice[3].append(inputjson['main_results'][0]['lstar'])
        
    if len(theta_max) == 2:
        # best_guesses_10ice[0].append(theta_max[0])
        # best_guesses_10ice[1].append(-3.5)
        # best_guesses_10ice[2].append(theta_max[1])
        # best_guesses_10ice[3].append(inputjson['main_results'][0]['lstar'])
        
        best_guesses_20ice_qfix[0].append(theta_max[0])
        best_guesses_20ice_qfix[1].append(-3.5)
        best_guesses_20ice_qfix[2].append(theta_max[1])
        best_guesses_20ice_qfix[4].append(np.percentile(flat_samples[:, 0], [16, 84]))
        best_guesses_20ice_qfix[5].append([0, 0])
        best_guesses_20ice_qfix[6].append(np.percentile(flat_samples[:, 1], [16, 84]))

        best_guesses_20ice_qfix[3].append(inputjson['main_results'][0]['lstar'])
        
best_guesses_20ice[4] = [[x[0] for x in best_guesses_20ice[4]], [x[1] for x in best_guesses_20ice[4]]]
best_guesses_20ice[5] = [[x[0] for x in best_guesses_20ice[5]], [x[1] for x in best_guesses_20ice[5]]]
best_guesses_20ice[6] = [[x[0] for x in best_guesses_20ice[6]], [x[1] for x in best_guesses_20ice[6]]]
best_guesses_20ice_qfix[4] = [[x[0] for x in best_guesses_20ice_qfix[4]], [x[1] for x in best_guesses_20ice_qfix[4]]]
best_guesses_20ice_qfix[5] = [[x[0] for x in best_guesses_20ice_qfix[5]], [x[1] for x in best_guesses_20ice_qfix[5]]]
best_guesses_20ice_qfix[6] = [[x[0] for x in best_guesses_20ice_qfix[6]], [x[1] for x in best_guesses_20ice_qfix[6]]]

for x in filelist_5ice:
    
    jsonname = x.split('_')[0]
    jsonfile = open("processed_data/{:}".format(jsonname + '.json'))
    inputjson = json.load(jsonfile)
    
    picklefile = open('samplers_5ice/{:}'.format(x), 'rb')
    sampler = dill.load(picklefile)
    samples = sampler.flatchain
    theta_max = samples[np.argmax(sampler.flatlnprobability)]
    flat_samples = sampler.get_chain(discard=100, flat=True)

    
    if len(theta_max) == 3:
        best_guesses_5ice[0].append(theta_max[0])
        best_guesses_5ice[1].append(theta_max[1])
        best_guesses_5ice[2].append(theta_max[2])
        best_guesses_5ice[4].append(np.percentile(flat_samples[:, 0], [16, 84]))
        best_guesses_5ice[5].append(np.percentile(flat_samples[:, 1], [16, 84]))
        best_guesses_5ice[6].append(np.percentile(flat_samples[:, 2], [16, 84]))

        best_guesses_5ice[3].append(inputjson['main_results'][0]['lstar'])
    
    if len(theta_max) == 2:
        # best_guesses_10ice[0].append(theta_max[0])
        # best_guesses_10ice[1].append(-3.5)
        # best_guesses_10ice[2].append(theta_max[1])
        # best_guesses_10ice[3].append(inputjson['main_results'][0]['lstar'])
        
        best_guesses_5ice_qfix[0].append(theta_max[0])
        best_guesses_5ice_qfix[1].append(-3.5)
        best_guesses_5ice_qfix[2].append(theta_max[1])
        best_guesses_5ice_qfix[4].append(np.percentile(flat_samples[:, 0], [16, 84]))
        best_guesses_5ice_qfix[5].append([0, 0])
        best_guesses_5ice_qfix[6].append(np.percentile(flat_samples[:, 1], [16, 84]))

        best_guesses_5ice_qfix[3].append(inputjson['main_results'][0]['lstar'])
        
best_guesses_5ice[4] = [[x[0] for x in best_guesses_5ice[4]], [x[1] for x in best_guesses_5ice[4]]]
best_guesses_5ice[5] = [[x[0] for x in best_guesses_5ice[5]], [x[1] for x in best_guesses_5ice[5]]]
best_guesses_5ice[6] = [[x[0] for x in best_guesses_5ice[6]], [x[1] for x in best_guesses_5ice[6]]]
best_guesses_5ice_qfix[4] = [[x[0] for x in best_guesses_5ice_qfix[4]], [x[1] for x in best_guesses_5ice_qfix[4]]]
best_guesses_5ice_qfix[5] = [[x[0] for x in best_guesses_5ice_qfix[5]], [x[1] for x in best_guesses_5ice_qfix[5]]]
best_guesses_5ice_qfix[6] = [[x[0] for x in best_guesses_5ice_qfix[6]], [x[1] for x in best_guesses_5ice_qfix[6]]]

# z = np.polyfit(best_guesses[0], best_guesses[1], 1)
# p = np.poly1d(z)
# z1 = np.polyfit(best_guesses[0], best_guesses[2], 1)
# p1 = np.poly1d(z1)
# z2 = np.polyfit(best_guesses[1], best_guesses[2], 1)
# p2 = np.poly1d(z2)

with open("weird_q.txt", "w") as output:
    for filename in weird_q:
        output.write(str(filename.split('_')[0]) + '\n')





# ##Parameters distribution

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# fig.set_size_inches(16, 9)
# fig.subplots_adjust(wspace=0.3)

# ax1.scatter(best_guesses[0], best_guesses[1], label='onepl', alpha = 0.7)
# ax1.scatter(best_guesses_gauss[0], best_guesses_gauss[1], label='gauss', alpha = 0.7)
# # ax1.scatter(best_guesses_10ice[0], best_guesses_10ice[1], label='10% ice')

# # ax1.plot(best_guesses[0],p(best_guesses[0]),"r--")
# ax1.legend()
# ax1.set(xlabel='mdust', ylabel='q')
# ax1.set_xscale('log')
# ax1.set_xlim([min(best_guesses[0])*0.1, max(best_guesses[0])*10])
# ax1.set_title('mdust vs. q')

# ax2.scatter(best_guesses[0], best_guesses[2], label='onepl', alpha = 0.7)
# ax2.scatter(best_guesses_gauss[0], best_guesses_gauss[2], label='gauss', alpha = 0.7)
# # ax2.scatter(best_guesses_10ice[0], best_guesses_10ice[2], label='10% ice')

# # ax2.plot(best_guesses[0],p1(best_guesses[0]),"r--")
# ax2.legend()
# ax2.set(xlabel='mdust', ylabel='amin')
# ax2.set_xscale('log')
# ax2.set_xlim([min(best_guesses[0])*0.1, max(best_guesses[0])*10])
# ax2.set_title('mdust vs. amin')

# ax3.scatter(best_guesses[1], best_guesses[2], label='onepl', alpha = 0.7)
# ax3.scatter(best_guesses_gauss[1], best_guesses_gauss[2], label='gauss', alpha = 0.7)
# # ax3.scatter(best_guesses[1], best_guesses[2], label='10% ice')

# # ax3.plot(best_guesses[1],p2(best_guesses[1]),"r--")
# ax3.legend()
# ax3.set(xlabel='q', ylabel='amin')
# ax3.set_xlim([min(best_guesses[1])-0.5, max(best_guesses[1])+0.5])
# ax3.set_title('q vs. amin')

# fig.savefig('distribution.png', dpi=1200)
# plt.close()





fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.set_size_inches(16, 9)
fig.subplots_adjust(wspace=0.3)

ax1.scatter(best_guesses_gauss[0], best_guesses_gauss[1], label='astrosil', alpha = 0.5)
ax1.scatter(best_guesses_10ice[0], best_guesses_10ice[1], label='10% ice', alpha = 0.5)
ax1.scatter(best_guesses_20ice[0], best_guesses_20ice[1], label='20% ice', alpha = 0.5)
ax1.scatter(best_guesses_5ice[0], best_guesses_5ice[1], label='5% ice', alpha = 0.5)
# ax1.errorbar(best_guesses_gauss[0], best_guesses_gauss[1], yerr=best_guesses_gauss[4], fmt="o", alpha = 0.8)
# ax1.errorbar(best_guesses_10ice[0], best_guesses_10ice[1], yerr=best_guesses_10ice[4], fmt="o", alpha = 0.8)
# ax1.errorbar(best_guesses_20ice[0], best_guesses_20ice[1], yerr=best_guesses_20ice[4], fmt="o", alpha = 0.8)
# ax1.errorbar(best_guesses_5ice[0], best_guesses_5ice[1], yerr=best_guesses_5ice[4], fmt="o", alpha = 0.8)

# ax1.plot(best_guesses[0],p(best_guesses[0]),"r--")
ax1.legend()
ax1.set(xlabel='mdust', ylabel='q')
ax1.set_xscale('log')
ax1.set_xlim([min(best_guesses_gauss[0])*0.1, max(best_guesses_gauss[0])*10])
ax1.set_title('mdust vs. q')

ax2.scatter(best_guesses_gauss[0], best_guesses_gauss[2], label='astrosil', alpha = 0.5)
ax2.scatter(best_guesses_10ice[0], best_guesses_10ice[2], label='10% ice', alpha = 0.5)
ax2.scatter(best_guesses_20ice[0], best_guesses_20ice[2], label='20% ice', alpha = 0.5)
ax2.scatter(best_guesses_5ice[0], best_guesses_5ice[2], label='5% ice', alpha = 0.5)
# ax2.errorbar(best_guesses_gauss[0], best_guesses_gauss[2], yerr=best_guesses_gauss[5], fmt="o", alpha = 0.8)
# ax2.errorbar(best_guesses_10ice[0], best_guesses_10ice[2], yerr=best_guesses_10ice[5], fmt="o", alpha = 0.8)
# ax2.errorbar(best_guesses_20ice[0], best_guesses_20ice[2], yerr=best_guesses_20ice[5], fmt="o", alpha = 0.8)
# ax2.errorbar(best_guesses_5ice[0], best_guesses_5ice[2], yerr=best_guesses_5ice[5], fmt="o", alpha = 0.8)

# ax2.plot(best_guesses[0],p1(best_guesses[0]),"r--")
ax2.legend()
ax2.set(xlabel='mdust', ylabel='amin')
ax2.set_xscale('log')
ax2.set_xlim([min(best_guesses_gauss[0])*0.1, max(best_guesses_gauss[0])*10])
ax2.set_title('mdust vs. amin')

ax3.scatter(best_guesses_gauss[1], best_guesses_gauss[2], label='astrosil', alpha = 0.5)
ax3.scatter(best_guesses_10ice[1], best_guesses_10ice[2], label='10% ice', alpha = 0.5)
ax3.scatter(best_guesses_20ice[1], best_guesses_20ice[2], label='20% ice', alpha = 0.5)
ax3.scatter(best_guesses_5ice[1], best_guesses_5ice[2], label='5% ice', alpha = 0.5)
# ax3.errorbar(best_guesses_gauss[1], best_guesses_gauss[2], yerr=best_guesses_gauss[6], fmt="o", alpha = 0.8)
# ax3.errorbar(best_guesses_10ice[1], best_guesses_10ice[2], yerr=best_guesses_10ice[6], fmt="o", alpha = 0.8)
# ax3.errorbar(best_guesses_20ice[1], best_guesses_20ice[2], yerr=best_guesses_20ice[6], fmt="o", alpha = 0.8)
# ax3.errorbar(best_guesses_5ice[1], best_guesses_5ice[2], yerr=best_guesses_5ice[6], fmt="o", alpha = 0.8)

# ax3.plot(best_guesses[1],p2(best_guesses[1]),"r--")
ax3.legend()
ax3.set(xlabel='q', ylabel='amin')
ax3.set_xlim([min(best_guesses_gauss[1])-0.5, max(best_guesses_gauss[1])+0.5])
ax3.set_title('q vs. amin')
fig.suptitle('Parameters Distribution')

fig.savefig('distribution_4runs.jpg', dpi=800)
plt.close()




# ##Parameters vs luminosity

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# fig.set_size_inches(16, 9)
# fig.subplots_adjust(wspace=0.3)

# ax1.scatter(best_guesses[3], best_guesses[1], label='onepl', alpha = 0.7)
# ax1.scatter(best_guesses_gauss[3], best_guesses_gauss[1], label='gauss', alpha = 0.7)

# # ax1.plot(best_guesses[0],p(best_guesses[0]),"r--")
# ax1.legend()
# ax1.set(xlabel='lstar', ylabel='q')
# ax1.set_xscale('log')
# ax1.set_xlim([min(best_guesses[3])*.1, max(best_guesses[3])*10])
# ax1.set_ylim([min(best_guesses[1])-0.5, max(best_guesses[1])+0.5])
# ax1.set_title('lstar vs. q')

# ax2.scatter(best_guesses[3], best_guesses[2], label='onepl', alpha = 0.7)
# ax2.scatter(best_guesses_gauss[3], best_guesses_gauss[2], label='gauss', alpha = 0.7)

# # ax2.plot(best_guesses[0],p1(best_guesses[0]),"r--")
# ax2.legend()
# ax2.set(xlabel='lstar', ylabel='amin')
# ax2.set_xscale('log')
# ax2.set_yscale('log')
# ax2.set_xlim([min(best_guesses[3])*.1, max(best_guesses[3])*10])
# ax2.set_ylim([min(best_guesses[2])*.9, max(best_guesses[2])*1.1])
# ax2.set_title('lstar vs. amin')

# ax3.scatter(best_guesses[3], best_guesses[0], label='onepl', alpha = 0.7)
# ax3.scatter(best_guesses_gauss[3], best_guesses_gauss[0], label='gauss', alpha = 0.7)

# # ax3.plot(best_guesses[1],p2(best_guesses[1]),"r--")
# ax3.legend()
# ax3.set(xlabel='lstar', ylabel='mdust')
# ax3.set_xscale('log')
# ax3.set_yscale('log')
# ax3.set_xlim([min(best_guesses[3])*.1, max(best_guesses[3])*10])
# ax3.set_ylim([min(best_guesses[0])*.5, max(best_guesses[0])*5])
# ax3.set_title('lstar vs. mdust')

# fig.savefig('distribution_lstar_gaussvsonepl.png', dpi=1200)
# plt.close()






fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.set_size_inches(16, 9)
fig.subplots_adjust(wspace=0.3)

ax1.scatter(best_guesses_gauss[3], best_guesses_gauss[1], label='astrosil', alpha = 0.7)
ax1.scatter(best_guesses_10ice[3], best_guesses_10ice[1], label='10% ice', alpha = 0.7)
ax1.scatter(best_guesses_20ice[3], best_guesses_20ice[1], label='20% ice', alpha = 0.5)
ax1.scatter(best_guesses_5ice[3], best_guesses_5ice[1], label='5% ice', alpha = 0.5)

# ax1.plot(best_guesses[0],p(best_guesses[0]),"r--")
ax1.legend()
ax1.set(xlabel='lstar', ylabel='q')
ax1.set_xscale('log')
ax1.set_xlim([min(best_guesses_gauss[3])*.1, max(best_guesses_gauss[3])*10])
ax1.set_ylim([min(best_guesses_gauss[1])-0.5, max(best_guesses_gauss[1])+0.5])
ax1.set_title('lstar vs. q')

ax2.scatter(best_guesses_gauss[3], best_guesses_gauss[2], label='astrosil', alpha = 0.7)
ax2.scatter(best_guesses_10ice[3], best_guesses_10ice[2], label='10% ice', alpha = 0.7)
ax2.scatter(best_guesses_20ice[3], best_guesses_20ice[2], label='20% ice', alpha = 0.5)
ax2.scatter(best_guesses_5ice[3], best_guesses_5ice[2], label='5% ice', alpha = 0.5)

# ax2.plot(best_guesses[0],p1(best_guesses[0]),"r--")
ax2.legend()
ax2.set(xlabel='lstar', ylabel='amin')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim([min(best_guesses_gauss[3])*.1, max(best_guesses_gauss[3])*10])
ax2.set_ylim([min(best_guesses_gauss[2])*.9, max(best_guesses_gauss[2])*1.1])
ax2.set_title('lstar vs. amin')

ax3.scatter(best_guesses_gauss[3], best_guesses_gauss[0], label='astrosil', alpha = 0.7)
ax3.scatter(best_guesses_10ice[3], best_guesses_10ice[0], label='10% ice', alpha = 0.7)
ax3.scatter(best_guesses_20ice[3], best_guesses_20ice[0], label='20% ice', alpha = 0.5)
ax3.scatter(best_guesses_5ice[3], best_guesses_5ice[0], label='5% ice', alpha = 0.5)

# ax3.plot(best_guesses[1],p2(best_guesses[1]),"r--")
ax3.legend()
ax3.set(xlabel='lstar', ylabel='mdust')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlim([min(best_guesses_gauss[3])*.1, max(best_guesses_gauss[3])*10])
ax3.set_ylim([min(best_guesses_gauss[0])*.5, max(best_guesses_gauss[0])*5])
ax3.set_title('lstar vs. mdust')
fig.suptitle('Parameters vs. luminosity')

fig.savefig('distribution_lstar_4runs.jpg', dpi=800)
plt.close()








fig, (ax2, ax3) = plt.subplots(1, 2)
fig.set_size_inches(16, 9)
fig.subplots_adjust(wspace=0.3)

ax2.scatter(best_guesses_gauss_qfix[3], best_guesses_gauss_qfix[2], label='astrosil', alpha = 0.7)
ax2.scatter(best_guesses_10ice_qfix[3], best_guesses_10ice_qfix[2], label='10% ice', alpha = 0.7)
ax2.scatter(best_guesses_20ice_qfix[3], best_guesses_20ice_qfix[2], label='20% ice', alpha = 0.5)
ax2.scatter(best_guesses_5ice_qfix[3], best_guesses_5ice_qfix[2], label='5% ice', alpha = 0.5)

# ax2.plot(best_guesses[0],p1(best_guesses[0]),"r--")
ax2.legend()
ax2.set(xlabel='lstar', ylabel='amin')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim([min(best_guesses_gauss[3])*.1, max(best_guesses_gauss[3])*10])
ax2.set_ylim([min(best_guesses_gauss[2])*.9, max(best_guesses_gauss[2])*1.1])
ax2.set_title('lstar vs. amin')

ax3.scatter(best_guesses_gauss_qfix[3], best_guesses_gauss_qfix[0], label='astrosil', alpha = 0.7)
ax3.scatter(best_guesses_10ice_qfix[3], best_guesses_10ice_qfix[0], label='10% ice', alpha = 0.7)
ax3.scatter(best_guesses_20ice_qfix[3], best_guesses_20ice_qfix[0], label='20% ice', alpha = 0.5)
ax3.scatter(best_guesses_5ice_qfix[3], best_guesses_5ice_qfix[0], label='5% ice', alpha = 0.5)

# ax3.plot(best_guesses[1],p2(best_guesses[1]),"r--")
ax3.legend()
ax3.set(xlabel='lstar', ylabel='mdust')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlim([min(best_guesses_gauss[3])*.1, max(best_guesses_gauss[3])*10])
ax3.set_ylim([min(best_guesses_gauss[0])*.5, max(best_guesses_gauss[0])*5])
ax3.set_title('lstar vs. mdust')
fig.suptitle('Parameters vs. luminosity fixed q')

fig.savefig('distribution_lstar_4runs_fixedq.jpg', dpi=800)
plt.close()






fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.set_size_inches(16, 9)
fig.subplots_adjust(wspace=0.3)

ax1.scatter(best_guesses_gauss_qfix[0], best_guesses_gauss_qfix[1], label='astrosil', alpha = 0.7)
ax1.scatter(best_guesses_10ice_qfix[0], best_guesses_10ice_qfix[1], label='10% ice', alpha = 0.7)
ax1.scatter(best_guesses_20ice_qfix[0], best_guesses_20ice_qfix[1], label='20% ice', alpha = 0.5)
ax1.scatter(best_guesses_5ice_qfix[0], best_guesses_5ice_qfix[1], label='5% ice', alpha = 0.5)

# ax1.plot(best_guesses[0],p(best_guesses[0]),"r--")
ax1.legend()
ax1.set(xlabel='mdust', ylabel='q')
ax1.set_xscale('log')
ax1.set_xlim([min(best_guesses_gauss[0])*0.1, max(best_guesses_gauss[0])*10])
ax1.set_title('mdust vs. q')

ax2.scatter(best_guesses_gauss_qfix[0], best_guesses_gauss_qfix[2], label='astrosil', alpha = 0.7)
ax2.scatter(best_guesses_10ice_qfix[0], best_guesses_10ice_qfix[2], label='10% ice', alpha = 0.7)
ax2.scatter(best_guesses_20ice_qfix[0], best_guesses_20ice_qfix[2], label='20% ice', alpha = 0.5)
ax2.scatter(best_guesses_5ice_qfix[0], best_guesses_5ice_qfix[2], label='5% ice', alpha = 0.5)

# ax2.plot(best_guesses[0],p1(best_guesses[0]),"r--")
ax2.legend()
ax2.set(xlabel='mdust', ylabel='amin')
ax2.set_xscale('log')
ax2.set_xlim([min(best_guesses_gauss[0])*0.1, max(best_guesses_gauss[0])*10])
ax2.set_title('mdust vs. amin')

ax3.scatter(best_guesses_gauss_qfix[1], best_guesses_gauss_qfix[2], label='astrosil', alpha = 0.7)
ax3.scatter(best_guesses_10ice_qfix[1], best_guesses_10ice_qfix[2], label='10% ice', alpha = 0.7)
ax3.scatter(best_guesses_20ice_qfix[1], best_guesses_20ice_qfix[2], label='20% ice', alpha = 0.5)
ax3.scatter(best_guesses_5ice_qfix[1], best_guesses_5ice_qfix[2], label='5% ice', alpha = 0.5)

# ax3.plot(best_guesses[1],p2(best_guesses[1]),"r--")
ax3.legend()
ax3.set(xlabel='q', ylabel='amin')
ax3.set_xlim([min(best_guesses_gauss[1])-0.5, max(best_guesses_gauss[1])+0.5])
ax3.set_title('q vs. amin')
fig.suptitle('Parameters distribution fixed q')

fig.savefig('distribution_4runs_fixedq.jpg', dpi=800)
plt.close()














fig, (ax1, ax2) = plt.subplots(2, 1)
fig.set_size_inches(7, 6)
fig.subplots_adjust(wspace=0.3)

ax1.scatter(best_guesses_gauss[3], best_guesses_gauss[1], label='astrosil', alpha = 0.7)
ax1.scatter(best_guesses_10ice[3], best_guesses_10ice[1], label='10% ice', alpha = 0.7)
ax1.scatter(best_guesses_20ice[3], best_guesses_20ice[1], label='20% ice', alpha = 0.5)
ax1.scatter(best_guesses_5ice[3], best_guesses_5ice[1], label='5% ice', alpha = 0.5)

# ax1.plot(best_guesses[0],p(best_guesses[0]),"r--")
ax1.legend()
ax1.set(ylabel='q')
ax1.set_xscale('log')
ax1.set_xlim([min(best_guesses_gauss[3])*.1, max(best_guesses_gauss[3])*10])
ax1.set_ylim(-4.25, -2.75)
ax1.invert_yaxis()
# ax1.set_title('lstar vs. q')

ax2.scatter(best_guesses_gauss[3], best_guesses_gauss[2], label='astrosil', alpha = 0.7)
ax2.scatter(best_guesses_10ice[3], best_guesses_10ice[2], label='10% ice', alpha = 0.7)
ax2.scatter(best_guesses_20ice[3], best_guesses_20ice[2], label='20% ice', alpha = 0.5)
ax2.scatter(best_guesses_5ice[3], best_guesses_5ice[2], label='5% ice', alpha = 0.5)

# ax2.plot(best_guesses[0],p1(best_guesses[0]),"r--")
ax2.legend()
ax2.set(xlabel='lstar', ylabel='amin')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim([min(best_guesses_gauss[3])*.1, max(best_guesses_gauss[3])*10])
ax2.set_ylim([min(best_guesses_gauss[2])*.9, max(best_guesses_gauss[2])*1.1])
# ax2.set_title('lstar vs. amin')

fig.suptitle('Parameters vs. luminosity')

fig.savefig('distribution_lstar_comp.jpg', dpi=800)
plt.close()
