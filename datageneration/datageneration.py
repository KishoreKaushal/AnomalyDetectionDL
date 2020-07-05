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

def main():
    seed = 324
    np.random.seed(seed)
    filename = "config.yaml"
    params = read_yaml(filename)
    numdaysinweek = params['inputs']['numdaysinweek']
    numtimebinsinday = params['inputs']['numtimebinsinday']
    enableapi = params['inputs']['enableapi']
    anamolyratio = params['inputs']['anamolyratio']

    savefile = params['inputs']['savefile']
    savepathforpkl = params['inputs']['savepathforpkl']
    savepathforcsv = params['inputs']['savepathforcsv']

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
                timebincount = 1
                for a2 in range(numdaysinweek):
                    # weekday
                    if a2 not in [5,6]:
                        numrequestsweekday = normaldistparams['numrequestsweekday']
                        requestsclustersizeweekday = normaldistparams['requestsclustersizeweekday']
                        apiclustersizeperbinweekday = normaldistparams['apiclustersizeperbinweekday']
                        
                        totrequests = numrequestsweekday + \
                                    np.random.randint(low=-int(0.01*numrequestsweekday),
                                                        high=int(0.01*numrequestsweekday)+1)
                        requestlist = list(totrequests*np.random.dirichlet(totrequests*np.array(requestsclustersizeweekday)))
                        # looping over bins in a day
                        for a3 in range(numtimebinsinday):
                            timebin = "BIN" + str(timebincount)
                            timebincount = timebincount + 1
                            if enableapi:
                                totrequestsperbin = requestlist[a3]
                                # considering cluster size of specific bin
                                requestlistperbin = list(totrequestsperbin*np.random.dirichlet(
                                                            totrequestsperbin*np.array(apiclustersizeperbinweekday[a3])))
                                for a4 in range(len(apiclustersizeperbinweekday[a3])):
                                    api = 'API'+str(a4)
                                    # anamoly calculation
                                    flag = np.random.binomial(1,anamolyratio)
                                    if flag:
                                        # as it an anamoly, select one of the avaliable distribution
                                        anamolydistnum = np.random.randint(low=1,high=len(allanamolydistparams)+1)
                                        anamolydistparams = allanamolydistparams[anamolydistnum]
                                        # variables
                                        if 'numrequestsweekday' in anamolydistparams:
                                            numrequestsweekdayanamoly = anamolydistparams['numrequestsweekday']
                                        else:
                                            numrequestsweekdayanamoly = normaldistparams['numrequestsweekday']
                                        if 'requestsclustersizeweekday' in anamolydistparams:
                                            requestsclustersizeweekdayanamoly = anamolydistparams['requestsclustersizeweekday']
                                        else:
                                            requestsclustersizeweekdayanamoly = normaldistparams['requestsclustersizeweekday']
                                        if 'apiclustersizeperbinweekday' in anamolydistparams:
                                            apiclustersizeperbinweekdayanamoly = anamolydistparams['apiclustersizeperbinweekday']
                                        else:
                                            apiclustersizeperbinweekdayanamoly = normaldistparams['apiclustersizeperbinweekday']
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
                                        datalist.append([apikey,api,timebin,numrequests,numfailures,label,anamolydistnum])
                                    else:
                                        numrequests = requestlistperbin[a4]
                                        numfailures = int(failurepercentage*numrequests/100)
                                        numrequests = int(numrequests)
                                        label = 'NOT ANAMOLY'
                                        datalist.append([apikey,api,timebin,numrequests,numfailures,label,np.nan])
                            else:
                                raise Exception("Not Implemented")
                    else: # weekend
                        numrequestsweekend = normaldistparams['numrequestsweekend']
                        requestsclustersizeweekend = normaldistparams['requestsclustersizeweekend']
                        apiclustersizeperbinweekend = normaldistparams['apiclustersizeperbinweekend']
                        
                        totrequests = numrequestsweekend + \
                                    np.random.randint(low=-int(0.01*numrequestsweekend),
                                                        high=int(0.01*numrequestsweekend)+1)
                        requestlist = list(totrequests*np.random.dirichlet(totrequests*np.array(requestsclustersizeweekend)))
                        # looping over bins in a day
                        for a3 in range(numtimebinsinday):
                            timebin = "BIN" + str(timebincount)
                            timebincount = timebincount + 1
                            if enableapi:
                                totrequestsperbin = requestlist[a3]
                                # considering cluster size of specific bin
                                requestlistperbin = list(totrequestsperbin*np.random.dirichlet(
                                                            totrequestsperbin*np.array(apiclustersizeperbinweekend[a3])))                        
                                for a4 in range(len(apiclustersizeperbinweekend[a3])):
                                    api = 'API'+str(a4)
                                    # anamoly calculation
                                    flag = np.random.binomial(1,anamolyratio)
                                    if flag:
                                        # as it an anamoly, select one of the avaliable distribution
                                        anamolydistnum = np.random.randint(low=1,high=len(allanamolydistparams)+1)
                                        anamolydistparams = allanamolydistparams[anamolydistnum]
                                        # variables
                                        if 'numrequestsweekend' in anamolydistparams:
                                            numrequestsweekendanamoly = anamolydistparams['numrequestsweekend']
                                        else:
                                            numrequestsweekendanamoly = normaldistparams['numrequestsweekend']
                                        if 'requestsclustersizeweekend' in anamolydistparams:    
                                            requestsclustersizeweekendanamoly = anamolydistparams['requestsclustersizeweekend']
                                        else:
                                            requestsclustersizeweekendanamoly = normaldistparams['requestsclustersizeweekend']
                                        if 'apiclustersizeperbinweekend' in anamolydistparams:    
                                            apiclustersizeperbinweekendanamoly = anamolydistparams['apiclustersizeperbinweekend']
                                        else:
                                            apiclustersizeperbinweekendanamoly = normaldistparams['apiclustersizeperbinweekend']
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
                                        datalist.append([apikey,api,timebin,numrequests,numfailures,label,anamolydistnum])
                                    else:
                                        numrequests = requestlistperbin[a4]
                                        numfailures = int(failurepercentage*numrequests/100)
                                        numrequests = int(numrequests)
                                        label = 'NOT ANAMOLY'
                                        datalist.append([apikey,api,timebin,numrequests,numfailures,label,np.nan])
                            else:
                                raise Exception("Not Implemented")

    if enableapi:
        columns=['APIKEY','API','TIMEBIN','NUMREQUESTS','NUMFAILURES','LABEL','ANAMOLYDISTNUM']
    else:
        raise Exception("Not Implemented")
        
    data = pd.DataFrame(datalist, columns=columns)
    print(data.shape)
    print(data.head(10))
    if savefile:
        data.to_pickle(savepathforpkl)
        data.to_csv(savepathforcsv, index=False)

if __name__ == "__main__":
    main()