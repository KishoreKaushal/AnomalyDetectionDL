import os
import yaml
import tqdm

import numpy as np
import pandas as pd

# for reading yaml file
def read_yaml(filename):
    with open(filename, 'r') as stream:
        try:
            f = yaml.safe_load(stream)
            return f
        except Exception as e:
            raise Exception("{} is not present. Check the path".format(filename))

def expandlist(l, newlen):
    if newlen < len(l):
        raise Exception("new length must be higher than old length")
    elif newlen==len(l):
        return l
    elif newlen%len(l)!=0:
        raise Exception("new length must be a multiple of old length")
    else:
        expansion = newlen//len(l)
        newlist = []
        for i in range(len(l)):
                if not isinstance(l[i],list):
                    newlist.extend([l[i]/expansion]*expansion)
                else:
                    newlist.extend([list(np.array(l[i])/expansion)]*expansion)
        return newlist

def main():
    seed = 324
    np.random.seed(seed)
    filename = "config.yaml"
    params = read_yaml(filename)
    numdays = params['inputs']['numdays']
    numtimebinsinday = params['inputs']['numtimebinsinday']
    enableapi = params['inputs']['enableapi']
    anamolyratio = params['inputs']['anamolyratio']

    savefile = params['inputs']['savefile']
    savepathforpkl = params['inputs']['savepathforpkl']
    savepathforcsv = params['inputs']['savepathforcsv']

    # day dict
    daydict = {0:"MON",1:"TUE",2:"WED",3:"THU",4:"FRI",5:"SAT",6:"SUN"}

    datalist = []
    apikeycount = 1
    for group in tqdm.tqdm(params['inputs']['normaldistributions'].keys()):
        for normaldistnum in params['inputs']['normaldistributions'][group].keys():
            normaldistparams = params['inputs']['normaldistributions'][group][normaldistnum]
            allanamolydistparams = params['inputs']['anamolydistributions'][group]

            numapikey = normaldistparams['numapikey']
            failurepercentage = normaldistparams['failurepercentage']

            for a1 in range(numapikey):
                apikey = "APIKEY"+str(apikeycount)
                apikeycount = apikeycount + 1
                # timebincount = 1
                for a2 in range(numdays):
                    # we will assume the day start from monday
                    # weekday
                    if a2%7 not in [5,6]:
                        numrequestsweekday = normaldistparams['numrequestsweekday']
                        requestsclustersizeweekday = expandlist(normaldistparams['requestsclustersizeweekday'], numtimebinsinday)
                        apiclustersizeperbinweekday = expandlist(normaldistparams['apiclustersizeperbinweekday'], numtimebinsinday)
                        
                        totrequests = numrequestsweekday + \
                                    np.random.randint(low=-int(0.01*numrequestsweekday),
                                                        high=int(0.01*numrequestsweekday)+1)
                        requestlist = list(totrequests*np.random.dirichlet(totrequests*np.array(requestsclustersizeweekday)))
                        # looping over bins in a day
                        for a3 in range(numtimebinsinday):
                            timebin = "BIN" + str(a3+1)
                            # timebincount = timebincount + 1
                            totrequestsperbin = requestlist[a3]
                            if enableapi:
                                # considering cluster size of specific bin
                                requestlistperbin = list(totrequestsperbin*np.random.dirichlet(
                                                            totrequestsperbin*np.array(apiclustersizeperbinweekday[a3])))
                                for a4 in range(len(apiclustersizeperbinweekday[a3])):
                                    api = 'API'+str(a4)
                                    # it's anamoly with probability "anamolyratio"
                                    flag = np.random.binomial(1,anamolyratio)
                                    if flag:
                                        # as it an anamoly, select one of the avaliable distribution
                                        anamolydistnum = np.random.randint(low=1,high=len(allanamolydistparams)+1)
                                        anamolydistparams = allanamolydistparams[anamolydistnum]
                                        # variables
                                        if anamolydistparams.get('numrequestsweekday'):
                                            numrequestsweekdayanamoly = (anamolydistparams['numrequestsweekday'] + np.random.randint(low=0,high=3))*normaldistparams['numrequestsweekday']
                                        else:
                                            numrequestsweekdayanamoly = normaldistparams['numrequestsweekday']
                                        if anamolydistparams.get('requestsclustersizeweekday'):
                                            requestsclustersizeweekdayanamoly = expandlist(anamolydistparams['requestsclustersizeweekday'], numtimebinsinday)
                                        else:
                                            requestsclustersizeweekdayanamoly = expandlist(normaldistparams['requestsclustersizeweekday'], numtimebinsinday)
                                        if anamolydistparams.get('apiclustersizeperbinweekday'):
                                            apiclustersizeperbinweekdayanamoly = expandlist(anamolydistparams['apiclustersizeperbinweekday'], numtimebinsinday)
                                        else:
                                            apiclustersizeperbinweekdayanamoly = expandlist(normaldistparams['apiclustersizeperbinweekday'], numtimebinsinday)
                                        # as it's an anamoly we have to calculate the number of requests from the start
                                        totrequestsanamoly = numrequestsweekdayanamoly + \
                                                                np.random.randint(low=-int(0.01*numrequestsweekdayanamoly),
                                                                                high=int(0.01*numrequestsweekdayanamoly)+1)
                                        requestlistanamoly = list(totrequestsanamoly*np.random.dirichlet(totrequestsanamoly*np.array(requestsclustersizeweekdayanamoly)))
                                        totrequestsperbinanamoly =  requestlistanamoly[a3]
                                        requestlistperbinanamoly =  list(totrequestsperbinanamoly*np.random.dirichlet(
                                                                            totrequestsperbinanamoly*np.array(apiclustersizeperbinweekdayanamoly[a3])))
                                        numrequests = requestlistperbinanamoly[a4]
                                        numfailures = int(failurepercentage*numrequests/100)
                                        numrequests = int(numrequests)
                                        label = 'ANAMOLY'
                                        datalist.append([apikey,api,daydict[a2%7],timebin,numrequests,numfailures,label,anamolydistnum])
                                    else:
                                        numrequests = requestlistperbin[a4]
                                        numfailures = int(failurepercentage*numrequests/100)
                                        numrequests = int(numrequests)
                                        label = 'NOT ANAMOLY'
                                        datalist.append([apikey,api,daydict[a2%7],timebin,numrequests,numfailures,label,np.nan])
                            else:
                                # api is not enabled
                                flag = np.random.binomial(1,anamolyratio)
                                if flag:
                                    # as it'a anamoly select one of the available distribution
                                    anamolydistnum = np.random.randint(low=1,high=len(allanamolydistparams)+1)
                                    anamolydistparams = allanamolydistparams[anamolydistnum]
                                    # variables
                                    if anamolydistparams.get('numrequestsweekday'):
                                        numrequestsweekdayanamoly = (anamolydistparams['numrequestsweekday'] + np.random.randint(low=0,high=3))*normaldistparams['numrequestsweekday']
                                    else:
                                        numrequestsweekdayanamoly = normaldistparams['numrequestsweekday']
                                    if anamolydistparams.get('requestsclustersizeweekday'):
                                        requestsclustersizeweekdayanamoly = expandlist(anamolydistparams['requestsclustersizeweekday'], numtimebinsinday)
                                    else:
                                        requestsclustersizeweekdayanamoly = expandlist(normaldistparams['requestsclustersizeweekday'], numtimebinsinday)
                                    # as it's an anamoly we have to calculate the number of requests from the start
                                    totrequestsanamoly = numrequestsweekdayanamoly + \
                                                            np.random.randint(low=-int(0.01*numrequestsweekdayanamoly),
                                                                            high=int(0.01*numrequestsweekdayanamoly)+1)
                                    requestlistanamoly = list(totrequestsanamoly*np.random.dirichlet(totrequestsanamoly*np.array(requestsclustersizeweekdayanamoly)))
                                    numrequests = requestlistanamoly[a3]
                                    numfailures = int(failurepercentage*numrequests/100)
                                    numrequests = int(numrequests)
                                    label = 'ANAMOLY'
                                    datalist.append([apikey,daydict[a2%7],timebin,numrequests,numfailures,label,anamolydistnum])
                                else:
                                    numrequests = requestlist[a3]
                                    numfailures = int(failurepercentage*numrequests/100)
                                    numrequests = int(numrequests)
                                    label = 'NOT ANAMOLY'
                                    datalist.append([apikey,daydict[a2%7],timebin,numrequests,numfailures,label,np.nan])
                    else: # weekend
                        numrequestsweekend = normaldistparams['numrequestsweekend']
                        requestsclustersizeweekend = expandlist(normaldistparams['requestsclustersizeweekend'], numtimebinsinday)
                        apiclustersizeperbinweekend = expandlist(normaldistparams['apiclustersizeperbinweekend'], numtimebinsinday)
                        
                        totrequests = numrequestsweekend + \
                                    np.random.randint(low=-int(0.01*numrequestsweekend),
                                                        high=int(0.01*numrequestsweekend)+1)
                        requestlist = list(totrequests*np.random.dirichlet(totrequests*np.array(requestsclustersizeweekend)))
                        # looping over bins in a day
                        for a3 in range(numtimebinsinday):
                            timebin = "BIN" + str(a3)
                            # timebincount = timebincount + 1
                            totrequestsperbin = requestlist[a3]
                            if enableapi:
                                # considering cluster size of specific bin
                                requestlistperbin = list(totrequestsperbin*np.random.dirichlet(
                                                            totrequestsperbin*np.array(apiclustersizeperbinweekend[a3])))                        
                                for a4 in range(len(apiclustersizeperbinweekend[a3])):
                                    api = 'API'+str(a4)
                                    # it's anamoly with probability "anamolyratio"
                                    flag = np.random.binomial(1,anamolyratio)
                                    if flag:
                                        # as it an anamoly, select one of the avaliable distribution
                                        anamolydistnum = np.random.randint(low=1,high=len(allanamolydistparams)+1)
                                        anamolydistparams = allanamolydistparams[anamolydistnum]
                                        # variables
                                        if anamolydistparams.get('numrequestsweekend'):
                                            numrequestsweekendanamoly = (anamolydistparams['numrequestsweekend'] + np.random.randint(low=0,high=3))*normaldistparams['numrequestsweekend']
                                        else:
                                            numrequestsweekendanamoly = normaldistparams['numrequestsweekend']
                                        if anamolydistparams.get('requestsclustersizeweekend'):
                                            requestsclustersizeweekendanamoly = expandlist(anamolydistparams['requestsclustersizeweekend'], numtimebinsinday)
                                        else:
                                            requestsclustersizeweekendanamoly = expandlist(normaldistparams['requestsclustersizeweekend'], numtimebinsinday)
                                        if anamolydistparams.get('apiclustersizeperbinweekend'):
                                            apiclustersizeperbinweekendanamoly = expandlist(anamolydistparams['apiclustersizeperbinweekend'], numtimebinsinday)
                                        else:
                                            apiclustersizeperbinweekendanamoly = expandlist(normaldistparams['apiclustersizeperbinweekend'], numtimebinsinday)
                                        # as it's an anamoly we have to calculate the number of requests from the start
                                        totrequestsanamoly = numrequestsweekendanamoly + \
                                                                np.random.randint(low=-int(0.01*numrequestsweekendanamoly),
                                                                                high=int(0.01*numrequestsweekendanamoly)+1)
                                        requestlistanamoly = list(totrequestsanamoly*np.random.dirichlet(totrequestsanamoly*np.array(requestsclustersizeweekendanamoly)))
                                        totrequestsperbinanamoly =  requestlistanamoly[a3]
                                        requestlistperbinanamoly =  list(totrequestsperbinanamoly*np.random.dirichlet(
                                                                            totrequestsperbinanamoly*np.array(apiclustersizeperbinweekendanamoly[a3])))
                                        numrequests = requestlistperbinanamoly[a4]
                                        numfailures = int(failurepercentage*numrequests/100)
                                        numrequests = int(numrequests)
                                        label = 'ANAMOLY'
                                        datalist.append([apikey,api,daydict[a2%7],timebin,numrequests,numfailures,label,anamolydistnum])
                                    else:
                                        numrequests = requestlistperbin[a4]
                                        numfailures = int(failurepercentage*numrequests/100)
                                        numrequests = int(numrequests)
                                        label = 'NOT ANAMOLY'
                                        datalist.append([apikey,api,daydict[a2%7],timebin,numrequests,numfailures,label,np.nan])
                            else:
                                # api is not enabled
                                flag = np.random.binomial(1,anamolyratio)
                                if flag:
                                    # as it'a anamoly select one of the available distribution
                                    anamolydistnum = np.random.randint(low=1,high=len(allanamolydistparams)+1)
                                    anamolydistparams = allanamolydistparams[anamolydistnum]
                                    # variables
                                    if anamolydistparams.get('numrequestsweekend'):
                                        numrequestsweekendanamoly = (anamolydistparams['numrequestsweekend'] + np.random.randint(low=0,high=3))*normaldistparams['numrequestsweekend']
                                    else:
                                        numrequestsweekendanamoly = normaldistparams['numrequestsweekend']
                                    if anamolydistparams.get('requestsclustersizeweekend'):    
                                        requestsclustersizeweekendanamoly = expandlist(anamolydistparams['requestsclustersizeweekend'], numtimebinsinday)
                                    else:
                                        requestsclustersizeweekendanamoly = expandlist(normaldistparams['requestsclustersizeweekend'], numtimebinsinday)
                                    # as it's an anamoly we have to calculate the number of requests from the start
                                    totrequestsanamoly = numrequestsweekendanamoly + \
                                                            np.random.randint(low=-int(0.01*numrequestsweekendanamoly),
                                                                            high=int(0.01*numrequestsweekendanamoly)+1)
                                    requestlistanamoly = list(totrequestsanamoly*np.random.dirichlet(totrequestsanamoly*np.array(requestsclustersizeweekendanamoly)))
                                    numrequests = requestlistanamoly[a3]
                                    numfailures = int(failurepercentage*numrequests/100)
                                    numrequests = int(numrequests)
                                    label = 'ANAMOLY'
                                    datalist.append([apikey,daydict[a2%7],timebin,numrequests,numfailures,label,anamolydistnum])
                                else:
                                    numrequests = requestlist[a3]
                                    numfailures = int(failurepercentage*numrequests/100)
                                    numrequests = int(numrequests)
                                    label = 'NOT ANAMOLY'
                                    datalist.append([apikey,daydict[a2%7],timebin,numrequests,numfailures,label,np.nan])

    if enableapi:
        columns=['APIKEY','API','DAY','TIMEBIN','NUMREQUESTS','NUMFAILURES','LABEL','ANAMOLYDISTNUM']
    else:
        columns = ['APIKEY','DAY','TIMEBIN','NUMREQUESTS','NUMFAILURES','LABEL','ANAMOLYDISTNUM']
        
    data = pd.DataFrame(datalist, columns=columns)
    print(data.shape)
    print(data.head(10))
    if savefile:
        data.to_pickle(savepathforpkl)
        data.to_csv(savepathforcsv, index=False)

if __name__ == "__main__":
    main()