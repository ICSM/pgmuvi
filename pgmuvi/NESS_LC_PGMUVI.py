#!pip install gpytorch
#!pip3 install pyro-ppl
#!pip install h5glance
from test_script import *
import h5py 
from h5glance import H5Glance
from astropy.table import Table
from matplotlib import pyplot as plt
from astropy.table import Table, join
import json
import os
from astropy.timeseries import LombScargle
from astropy.table import Table, MaskedColumn


with open('/Users/d.vasquez/Desktop/NESS_2023/tescope_filters.json', 'r') as fp:
    data = json.load(fp)
patt_lc='/Users/d.vasquez/Desktop/NESS_2023/data_h5/light_curves.h5'
f = h5py.File(patt_lc, "r+",)
"""
path_ssb='04573-1452/WHITELOCK/GLASS_H'
t = Table.read(patt_lc, path=path_ssb) 
frequency, power = LombScargle(t['JD'], t['Magnitude']).autopower()
plt.plot(frequency, power)
plt.show()
per,mag,del_per=run_pgmuvi(LCfile =patt_lc,path_table= path_ssb,yerr=False)
#print(per,mag,'!',del_per,'!')
"""


table_per=Table()
stars=list(f.keys())
table_per.add_column(stars,name='IRASPSC')
#empy_col=np.full_like(stars, 0.0, dtype=np.double)
empy_col=np.zeros(len(stars))
for tel, bands in data.items():
   print(tel,":")
   for bn,filter in bands.items():
     table_per.add_column(empy_col,name=str('mag_'+tel+'_'+filter))
     table_per.add_column(empy_col,name=str('per_'+tel+'_'+filter))
     table_per.add_column(empy_col,name=str('err_per_'+tel+'_'+filter))
     print("\t➜",bn,":",filter)

def path_lc(file,dic_surv,data_per):
  list_paht=[]
  for star in list(file.keys()):
    #try:  
    for survery, band in dic_surv.items():
        #try:
        for bn,filter in band.items():
          try:
            path_ssb=str(star+'/'+survery+'/'+filter)
            print(path_ssb,'\t',file[star][survery][filter].shape[0])
            
            t = Table.read(patt_lc, path=path_ssb)
            
            if 'Magnitude_err' in t.colnames:
              per,mag,del_per=run_pgmuvi(LCfile =patt_lc,path_table= path_ssb,yerr=True,per_guess=400,plot_graph=False,save_plot=True,num_tra_iter=300,tra_mag=True)  
            else:   
              per,mag,del_per=run_pgmuvi(LCfile =patt_lc,path_table= path_ssb,per_guess=400,plot_graph=False,save_plot=True,num_tra_iter=300,tra_mag=True) 
            #per=2.0
            #mag=1.0
            #del_per=3.0
            list_paht.append(str(star+'/'+survery+'/'+filter))
            data_per[str('mag_'+survery+'_'+filter)][table_per['IRASPSC']==star]=mag
            data_per[str('per_'+survery+'_'+filter)][table_per['IRASPSC']==star]=per
            data_per[str('err_per_'+survery+'_'+filter)][table_per['IRASPSC']==star]=del_per
          except:
            pass  
      #data=file[star][survery][band]
    #except:
    #  print('does not exist')
  return list_paht  


listaaaa=path_lc(f,data,table_per)

for col_n in table_per.colnames:
  table_per[col_n] = MaskedColumn(data = table_per[col_n], mask = table_per[col_n]==0.0)


table_per.write('data_per_NESS_v2.csv',overwrite=True)  



"""

path_ssb='01085+3022/GAIA_DR3/GAIA_GAIA3.Gbp'#'01085+3022/PANSTARRS/PAN-STARRS_PS1.r'#
t = Table.read(patt_lc, path=path_ssb)
print(t.colnames)

if 'Magnitude_err' in t.colnames:
  per,mag,del_per=run_pgmuvi(LCfile =patt_lc,path_table= path_ssb,yerr=True)  
  print(per,mag,del_per)
  print((del_per/per)*100)
else:   
  per,mag,del_per=run_pgmuvi(LCfile =patt_lc,path_table= path_ssb) 
  print(per,mag,del_per) 


table_per=Table()
stars=list(f.keys())
table_per.add_column(stars,name='IRASPSC')
#empy_col=np.full_like(stars, 0.0, dtype=np.double)
empy_col=np.zeros(len(stars))

"""
"""

with open('/Users/d.vasquez/Desktop/NESS_2023/tescope_filters.json', 'r') as fp:
    data = json.load(fp)

f = h5py.File("/Users/d.vasquez/Desktop/NESS_2023/data_h5/light_curves.h5", "r+",)

table_per=Table()
stars=list(f.keys())
table_per.add_column(stars,name='IRASPSC')
empy_col=np.empty_like(stars)

#table_per.add_column(empy_col,name='survery')

for tel, bands in data.items():
   print(tel,":")
   for bn,filter in bands.items():
     table_per.add_column(empy_col,name=str('mag_'+tel+'_'+filter))
     table_per.add_column(empy_col,name=str('per_'+tel+'_'+filter))
     table_per.add_column(empy_col,name=str('err_per_'+tel+'_'+filter))
     print("\t➜",bn,":",filter)

table_per.pprint()

def path_lc(file,dic_surv,data_per):
  list_paht=[]
  for star in list(file.keys()):
    try:  
      for survery, band in dic_surv.items():
        try:
          for bn,filter in band.items():
            print(str(star+'/'+survery+'/'+filter),'\t',file[star][survery][filter].shape[0])
            list_paht.append(str(star+'/'+survery+'/'+filter))
            data_per[str('mag_'+survery+'_'+filter)][table_per['IRASPSC']==star]='mag'
            data_per[str('per_'+survery+'_'+filter)][table_per['IRASPSC']==star]='per'
            data_per[str('err_per_'+survery+'_'+filter)][table_per['IRASPSC']==star]='err_per'
        except:
          pass  
      #data=file[star][survery][band]
    except:
      print('does not exist')
  return list_paht

listaaaa=path_lc(f,data,table_per)

print(len(listaaaa))

table_per.pprint_all()

for tel, bands in data.items():
   print(tel,":")
   for bn,filter in bands.items():
     print("\t➜",bn,":",filter)

table_per.write('periodos.csv',overwrite=True)


"""
"""

N_D_TABLE=Table.read("/Users/d.vasquez/Desktop/NESS_2023/DIRBE_NEOWISE_NESS_TABLE.csv",format = 'csv')

def filter_path(stars,survey,band):
  list_path=[]
  for s in stars:
    list_path.append('/'+s+'/'+survey+'/'+band)
  return list_path

list_DIRBE=filter_path(N_D_TABLE['IRASPSC_1'],'DIRBE','COBE_DIRBE.3p5m')
list_DIRBE_2p2=filter_path(N_D_TABLE['IRASPSC_1'],'DIRBE','COBE_DIRBE.2p2m')
list_NEO=filter_path(N_D_TABLE['IRASPSC_1'],'NEOWISE','WISE_WISE.W1')

#list_NEO

#/00192-2020/DIRBE/COBE_DIRBE.1p25m

per_DIRBE=[]
del_per_DIRBE=[]
mag_DIRBE=[]
porcen_DIRBE=[]
for path_D in list_DIRBE:
  try:
    per,del_per,mag=run_pgmuvi(LCfile ='/content/light_curves.h5',path_table=path_D)
    mag_DIRBE.append(mag)
    del_per_DIRBE.append(del_per)
    per_DIRBE.append(per) 
    porcen_DIRBE.append((del_per/per)*100)
  except:
    mag_DIRBE.append('')
    del_per_DIRBE.append('')
    per_DIRBE.append('')
    porcen_DIRBE.append('')

Table_pgmuvi=Table([N_D_TABLE['IRASPSC_1'],mag_DIRBE,per_DIRBE,del_per_DIRBE,porcen_DIRBE,],names=('IRASPSC','mag_3p5','per_3p5','del_per_3.5','del_p_100'))

Table_pgmuvi.pprint_all()

per_NEO=[]
mag_NEO=[]
del_per_NEO=[]
porcen_NEO=[]
for path_N in list_NEO:
  try:
    per,del_per,mag=run_pgmuvi(LCfile ='/Users/d.vasquez/Desktop/NESS_2023/data_h5/light_curves.h5',path_table=path_N)
    per_NEO.append(per)
    mag_NEO.append(mag)
    del_per_NEO.append(del_per)
    porcen_NEO.append((del_per/per)*100)
  except:
    per_NEO.append()
    mag_NEO.append()
    del_per_NEO.append()
    porcen_NEO.append()
"""
"""
Table_pgmuvi=Table([N_D_TABLE['IRASPSC_1'],mag_NEO,per_NEO,del_per_NEO,porcen_NEO,],names=('IRASPSC','mag_w1','per_w1','del_per_w1','del_p_100_w1'))

Table_pgmuvi.pprint_all()

per_2p2_DIRBE=[]
mag_2p2_DIRBE=[]
for path_D in list_DIRBE_2p2:
  try:
    per,mag=run_pgmuvi(LCfile ='/content/light_curves.h5',path_table=path_D)
    mag_2p2_DIRBE.append(mag)
    per_2p2_DIRBE.append(per) 
  except:
    mag_2p2_DIRBE.append('')
    per_2p2_DIRBE.append('')





Table_pgmuvi=Table([N_D_TABLE['IRASPSC_1'],N_D_TABLE['DIRBE_ID'],mag_2p2_DIRBE,per_2p2_DIRBE,mag_DIRBE,per_DIRBE,N_D_TABLE['AllWISE_ID'],mag_NEO,per_NEO,N_D_TABLE['GaiaDR2_ID'],N_D_TABLE['AllWISE_TMASS_KEY'],],names=('IRASPSC','DIRBE_ID','mag_2p2','per_2p2','mag_3p5','per_3p5','AllWISE_ID','mag_w1','per_w1','GaiaDR2_ID','AllWISE_TMASS_KEY'))

Table_pgmuvi.write('PGMUVI_PER_MAG_DIRBE_NEOWISE_NESS.csv',overwrite=True)

"""

#run_pgmuvi(LCfile ='/content/light_curves.h5',path_table=per_mag_DIRBE[2])

#per,mag=run_pgmuvi(LCfile ='/content/light_curves.h5',path_table= '/15030-5319/GAIA_DR3/GAIA_GAIA3.G')

#print(per,mag)

#f = h5py.File("/content/light_curves.h5", "r+")

#H5Glance(f)

#f.close()

#t.pprint_all(),len(t)

#t2 = Table.read('/content/light_curves.h5', path='/18397+1738/GAIA_DR3/GAIA_GAIA3.Grp')

#plt.scatter(t[t.colnames[0]],t[t.colnames[1]])
#plt.ylabel(r"$mag$")
#plt.xlabel(r"$HJD$")

#run_pgmuvi(LCfile = '/content/1_Whitelock_R_Lep.csv', timecolumn = 'JD-2400000', magcolumn = '\tmag')

#run_pgmuvi(LCfile ='/content/light_curves_w_nwise_G_ASN_APP10.h5',path_table= '18397+1738/NEOWISE/w1')

#run_pgmuvi(LCfile ='/content/light_curves_w_nwise_G_ASN_APP10.h5',path_table= '18397+1738/Whitelock/Lmag')

