# coding: utf-8
"""
  # #!/usr/bin/env python
  See Peter's code here:
      https://github.com/Peter-Metz/state_taxdata/blob/master/state_taxdata/prepdata.py

  List of official puf files:
      https://docs.google.com/document/d/1tdo81DKSQVee13jzyJ52afd9oR68IwLpYZiXped_AbQ/edit?usp=sharing
      Per Peter latest file is here (8/20/2020 as of 9/13/2020)
      https://www.dropbox.com/s/hyhalpiczay98gz/puf.csv?dl=0
      C:\Users\donbo\Dropbox (Personal)\PUF files\files_based_on_puf2011\2020-08-20
      # raw string allows Windows-style slashes
      # r'C:\Users\donbo\Downloads\taxdata_stuff\puf_2017_djb.csv'

https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

@author: donbo
"""

# %% imports
import taxcalc as tc
import pandas as pd
import numpy as np
from bokeh.io import show, output_notebook

import src.reweight as rw

from timeit import default_timer as timer


# %% locations and file names
DATADIR = 'C:/programs_python/weighting/puf/data/'
HDFDIR = 'C:/programs_python/weighting/puf/ignore/'
BASE_NAME = 'puf_adjusted'
PUF_HDF = HDFDIR + BASE_NAME + '.h5'  # hdf5 is lightning fast


# %% constants
# agi stubs
# AGI groups to target separately
IRS_AGI_STUBS = [-9e99, 1.0, 5e3, 10e3, 15e3, 20e3, 25e3, 30e3, 40e3, 50e3,
                 75e3, 100e3, 200e3, 500e3, 1e6, 1.5e6, 2e6, 5e6, 10e6, 9e99]
HT2_AGI_STUBS = [-9e99, 1.0, 10e3, 25e3, 50e3, 75e3, 100e3,
                 200e3, 500e3, 1e6, 9e99]

# dictionary xwalks between target name and puf name, AFTER constructing
# variables as needed in targets and in puf (as noted below)
TARGPUF_XWALK = dict(nret_all='nret_all',  # Table 1.1
                     # puf create nret_mars2, nret_mars1
                     nret_mfjss='nret_mars2',  # Table 1.2
                     nret_single='nret_mars1',  # Table 1.2
                     agi='c00100',  # Table 1.1
                     wages='e00200',  # Table 1.4
                     taxint='e00300',  # Table 1.4
                     orddiv='e00600',  # Table 1.4
                     # target cgnet = cggross - cgloss   # Table 1.4
                     cgnet='c01000',  # create cgnet in targets
                     # puf irapentot = e01400 + e01500 (taxable)
                     irapentot='irapentot',  # irapentot create in puf
                     socsectot='e02400',  # Table 1.4 NOTE THAT this is 'e'
                     ti='c04800'  # Table 1.1
                     )
TARGPUF_XWALK
# CAUTION: reverse xwalk relies on having only one keyword per value
PUFTARG_XWALK = {val: kw for kw, val in TARGPUF_XWALK.items()}


# %% get target data and check them
IRSDAT = DATADIR + 'targets2018.csv'
irstot = pd.read_csv(IRSDAT)
irstot

# get irsstub and incrange mapping
# create a uniform incrange variable
irstot.loc[irstot['irsstub'] == 0, 'incrange'] = 'All returns'
irstot.loc[irstot['irsstub'] == 1, 'incrange'] = 'No adjusted gross income'


incmap = irstot[['irsstub', 'incrange']].drop_duplicates()
incmap

# drop targets for which I haven't yet set column descriptions
irstot = irstot.dropna(axis=0, subset=['column_description'])
irstot
irstot.columns

# check counts
irstot[['src', 'variable', 'value']].groupby(['src', 'variable']).agg(['count'])
irstot[['variable', 'value']].groupby(['variable']).agg(['count'])  # unique list

# quick check to make sure duplicate variables have same values
check = irstot[irstot.irsstub == 0][['src', 'variable']]
idups = check.duplicated(subset='variable', keep=False)
check[idups].sort_values(['variable', 'src'])
dupvars = check[idups]['variable'].unique()
dupvars

# now check values
keep = (irstot.variable.isin(dupvars)) & (irstot.irsstub == 0)
dups = irstot[keep][['variable', 'src', 'column_description', 'value']]
dups.sort_values(['variable', 'src'])
# looks ok except for very minor differences - any target version should do


# %% prepare potential targets based on xwalks above

tlist = list(TARGPUF_XWALK.keys()) # these are the target variables we need to have
# need:
#    cgnet = cggross - cgloss   # Table 1.4
tlist.remove('cgnet')
tlist.append('cggross')
tlist.append('cgloss')
tlist

# get the proper data
irstot
# first, get all needed variables, then filter duplicates
df = irstot[irstot['variable'].isin(tlist)]
df[['variable', 'value']].groupby(['variable']).agg(['count'])
# agi, nret_all, and ti have duplicate values -- keep just src 18in11si.xls

keep1 = df['src'] == '18in11si.xls'
keep2 = np.logical_not(df['variable'].isin(['nret_all', 'agi', 'ti']))
keep = keep1 | keep2  # CAUTION: | is not the same as or
keep1.sum()
keep2.sum()
keep.sum()
target_base = df[keep][['variable', 'irsstub', 'value']]
target_base[['variable', 'value']].groupby(['variable']).agg(['count'])
# good, this is what we want

wide = target_base.pivot(index='irsstub', columns='variable', values='value')
wide['cgnet'] = wide['cggross'] - wide['cgloss']
wide = wide.drop(['cggross', 'cgloss'], axis=1)
wide['irsstub'] = wide.index
wide.columns
targets_long = pd.melt(wide, id_vars=['irsstub'])
targets_long['variable'].value_counts()

# put dollar-valued targets in dollars rather than thousands
condition = np.logical_not(targets_long['variable'].isin(['nret_all', 'nret_mfjss', 'nret_single']))
condition.sum()
# here is the numpy equivalent to R ifelse
targets_long['value'] = np.where(condition, targets_long['value'] * 1000, targets_long['value'])



# %% get advanced file
%time puf_2018 = pd.read_hdf(PUF_HDF)  # 1 sec
puf_2018.tail()
puf_2018.columns.sort_values().tolist()  # show all column names

pufsub = puf_2018.copy()  # new data frame
pufsub = pufsub.loc[pufsub["data_source"] == 1]  # ~7k records dropped
pufsub['IRS_STUB'] = pd.cut(
    pufsub['c00100'],
    IRS_AGI_STUBS,
    labels=list(range(1, len(IRS_AGI_STUBS))),
    right=False)
pufsub.columns.sort_values().tolist()  # show all column names


# %% get just the variables we want and create new needed variables
plist = list(PUFTARG_XWALK.keys()) # these are the target variables we need to have
plist
plist.append('pid')
plist.append('IRS_STUB')
plist.append('s006')
plist.append('MARS')
plist.remove('nret_all')  # create as 1
plist.remove('nret_mars1')  # MARS==1
plist.remove('nret_mars2')  # MARS==2
plist.append('e01400')
plist.append('e01500')
plist.remove('irapentot')  # e01400 + e01500
plist

# is everything from plist in pufsub.columns?
[x for x in plist if x not in pufsub.columns]  # yes, all set

pufbase = pufsub[plist].copy()
# pufbase = pufbase.rename(columns={"s006": "nret_all"})
pufbase['nret_all'] = 1
pufbase['nret_mars1'] = (pufbase['MARS'] == 1).astype(int)
pufbase['nret_mars2'] = (pufbase['MARS'] == 2).astype(int)
# verify
pufbase[['MARS', 'pid']].groupby(['MARS']).agg(['count'])
pufbase.nret_mars1.sum()
pufbase.nret_mars2.sum()
# all good
pufbase['irapentot'] = pufbase['e01400'] + pufbase['e01500']
pufbase = pufbase.drop(['MARS', 'e01400', 'e01500'], axis=1)
pufbase
pufbase.columns
# reorder columns
idvars = ['pid', 'IRS_STUB', 's006']
targvars = ['nret_all', 'nret_mars1', 'nret_mars2',
            'c00100', 'e00200', 'e00300', 'e00600',
            'c01000', 'e02400', 'c04800', 'irapentot']
pufcols = idvars + targvars
pufcols
pufbase = pufbase[pufcols]
pufbase.columns


# %% prepare a puf summary for potential target variables

def wsum(grp, sumvars, wtvar):
    return grp[sumvars].multiply(grp[wtvar], axis=0).sum()


pufsums = pufbase.groupby('IRS_STUB').apply(wsum,
                              sumvars=targvars,
                              wtvar='s006')

pufsums = pufsums.append(pufsums.sum().rename(0)).sort_values('IRS_STUB')
pufsums['irsstub'] = pufsums.index
pufsums

pufsums = pufsums.rename(columns=PUFTARG_XWALK)

pufsums_long = pd.melt(pufsums, id_vars=['irsstub'])


# %% combine IRS totals and PUF totals and compare
targets_long
pufsums_long

irscomp = targets_long
irscomp = irscomp.rename(columns={'value': 'irs'})
# irscomp['irs'] = pd.Series.astype(irscomp['irs'], 'float')
irscomp
irscomp.info()

pufcomp = pufsums_long
pufcomp = pufcomp.rename(columns={'value': 'puf'})
pufcomp
pufcomp.info()

comp = pd.merge(irscomp, pufcomp, on=['irsstub', 'variable'])
comp['diff'] = comp['puf'] - comp['irs']
comp['pdiff'] = comp['diff'] / comp['irs'] * 100
format_mapping = {'irs': '{:,.0f}',
                  'puf': '{:,.0f}',
                  'diff': '{:,.0f}',
                  'pdiff': '{:,.1f}'}
for key, value in format_mapping.items():
    comp[key] = comp[key].apply(value.format)

# comp['diffpctagi'] = comp['diff'] / comp[[('irsstub'==0)]]['irs']
# comp.pdiff = comp.pdiff.round(decimals=1)
comp

comp[(comp['variable'] == 'nret_all')]
comp[(comp['variable'] == 'agi')]
comp[(comp['variable'] == 'wages')]


# compshow = comp[(comp['variable'] == 'nret_all')]
# compshow.style.format({"irs": "${:20,.0f}"})

# pd.options.display.float_format = '{:,.1f}'.format
# pd.reset_option('display.float_format')


# %% target an income range
pufbase

pufbase.head()
pufbase.info()
pufbase.IRS_STUB.count()
pufbase.IRS_STUB.value_counts()
# pufbase['IRS_STUB'].value_counts()


def constraints(x, wh, xmat):
    return np.dot(x * wh, xmat)


targets_long


# prepare all targets
targets_all = irscomp.pivot(index='irsstub', columns='variable', values='irs')
targets_all = targets_all.rename(columns=TARGPUF_XWALK)
targets_all['IRS_STUB'] = targets_all.index
targets_all.columns

# prepare data
targcols = ['nret_all', 'c00100', 'e00200']
targcols = ['nret_all', 'nret_mars2', 'nret_mars1',
            'c00100', 'e00200', 'e00300', 'e00600',
            'irapentot', 'c01000', 'e02400', 'c04800']

targcols = ['nret_all', 'nret_mars2', 'nret_mars1',
            'c00100', 'e00200', 'e00300', 'e00600',
            'irapentot', 'c01000', 'e02400']

stub = 2
pufstub = pufbase.loc[pufbase['IRS_STUB'] ==  stub]

xmat = np.asarray(pufstub[targcols], dtype=float)
xmat.shape

wh = np.asarray(pufstub.s006)
targets_all.loc[targets_all['IRS_STUB'] == stub]
targets_stub = targets_all[targcols].loc[targets_all['IRS_STUB'] == stub]
targets_stub = np.asarray(targets_stub, dtype=float).flatten()

x0 = np.ones(wh.size)

# comp
t0 = constraints(x0, wh, xmat)
pdiff0 = t0 / targets_stub * 100 - 100
pdiff0


rwp = rw.Reweight(wh, xmat, targets_stub)
x, info = rwp.reweight(xlb=0.1, xub=10,
                       crange=.0001,
                       ccgoal=10, objgoal=100,
                       max_iter=50)
info['status_msg']

np.quantile(x, [0, .1, .25, .5, .75, .9, 1])

t1 = constraints(x, wh, xmat)
pdiff1 = t1 / targets_stub * 100 - 100
pdiff1


# %% loop through puf

def func(df):
    print(df.name)
    stub = df.name
    # pufstub = pufbase.loc[pufbase['IRS_STUB'] ==  stub]
    xmat = np.asarray(df[targcols], dtype=float)
    wh = np.asarray(df.s006)

    targets_all.loc[targets_all['IRS_STUB'] == stub]
    targets_stub = targets_all[targcols].loc[targets_all['IRS_STUB'] == stub]
    targets_stub = np.asarray(targets_stub, dtype=float).flatten()

    x0 = np.ones(wh.size)

    rwp = rw.Reweight(wh, xmat, targets_stub)
    x, info = rwp.reweight(xlb=0.1, xub=10,
                           crange=.0001,
                           ccgoal=10, objgoal=100,
                           max_iter=50)
    print(info['status_msg'])

    df['x'] = x
    return df

# targcols = ['nret_all', 'c00100', 'e00200']
alltargs = ['nret_all', 'nret_mars2', 'nret_mars1',
            'c00100', 'e00200', 'e00300', 'e00600',
            'irapentot', 'c01000', 'e02400', 'c04800']

targcols = ['nret_all', 'nret_mars2', 'nret_mars1',
            'c00100', 'e00200', 'e00300', 'e00600',
            'irapentot', 'c01000', 'e02400']

grouped = pufbase.groupby('IRS_STUB')
temp = pufbase.loc[pufbase['IRS_STUB'] == 1]

a = timer()
dfnew = grouped.apply(func)
# dfnew = temp.groupby('IRS_STUB').apply(lambda x: func(x, targcols))
b = timer()
b - a

dfnew
dfnew['wtnew'] = dfnew.s006 * dfnew.x


# %% examine results
targets_long
pufsums_long


result_sums = dfnew.groupby('IRS_STUB').apply(wsum,
                                               sumvars=alltargs,
                                               wtvar='wtnew')

result_sums = result_sums.append(result_sums.sum().rename(0)).sort_values('IRS_STUB')
result_sums
result_sums['irsstub'] = result_sums.index
result_sums = result_sums.rename(columns=PUFTARG_XWALK)
result_sums.columns

resultsums_long = pd.melt(result_sums, id_vars=['irsstub'])

resultscomp = resultsums_long
resultscomp = resultscomp.rename(columns={'value': 'pufrw'})
resultscomp
resultscomp.info()

# combine and format
comp2 = pd.merge(incmap, irscomp, on='irsstub')
comp2 = pd.merge(comp2, pufcomp, on=['irsstub', 'variable'])
comp2 = pd.merge(comp2, resultscomp, on=['irsstub', 'variable'])
comp2['puf_diff'] = comp2['puf'] - comp2['irs']
comp2['pufrw_diff'] = comp2['pufrw'] - comp2['irs']
comp2['puf_pdiff'] = comp2['puf_diff'] / comp2['irs'] * 100
comp2['pufrw_pdiff'] = comp2['pufrw_diff'] / comp2['irs'] * 100

# format the data
# mcols = list(comp2.columns)
# mcols[2:7]
comp3 = comp2.copy()
condition = np.logical_not(comp3['variable'].isin(['nret_all', 'nret_mfjss', 'nret_single']))
condition.sum()
changevars = ['irs', 'puf', 'pufrw', 'puf_diff', 'pufrw_diff']
# put dollar-valued items in $ millions
for var in changevars:
    comp3[var] = np.where(condition, comp3[var] / 1e9, comp3[var])

# CAUTION: This formatting creates strings!
format_mapping = {'irs': '{:,.0f}',
                  'puf': '{:,.0f}',
                  'pufrw': '{:,.0f}',
                  'puf_diff': '{:,.1f}',
                  'pufrw_diff': '{:,.1f}',
                  'puf_pdiff': '{:,.1f}',
                  'pufrw_pdiff': '{:,.1f}'}

for key, value in format_mapping.items():
    comp3[key] = comp3[key].apply(value.format)

dollarvars = ['irsstub', 'incrange', 'variable',
              'irs', 'puf', 'pufrw', 'puf_diff', 'pufrw_diff']

pctvars = ['irsstub', 'incrange', 'variable',
              'irs', 'puf', 'pufrw', 'puf_pdiff', 'pufrw_pdiff']

allvars = ['irsstub', 'incrange', 'variable',
           'irs', 'puf', 'pufrw', 'puf_diff', 'pufrw_diff',
           'puf_pdiff', 'pufrw_pdiff']


# pd.options.display.max_columns = 8
# pd.reset_option('display.max_columns')
comp3['variable'].value_counts()
var = 'nret_all'
var = 'nret_mfjss'
var = 'nret_single'
var = 'agi'
var = 'irapentot'
var = 'taxint'
var = 'cgnet'
var = 'ti'
comp3.loc[comp3['variable'] == var, dollarvars]
comp3.loc[comp3['variable'] == var, pctvars]
# comp3.loc[comp3['variable'] == var, allvars]




# %% old stuff
stub = df.loc[df['IRS_STUB'] == 3].copy()
stub['ones'] = 1.0


tcols = ['nret_all', 'agi', 'wages']
xcols = ['nret', 'c00100', 'e00200']
targets_stub = targets_all[tcols].iloc[stub]
targets_stub = np.asarray(targets_stub, dtype=float)

targets = irstot[cols].iloc[3]
targets.agi = targets.agi * 1000.
targets = np.asarray(targets, dtype=float)
type(targets)
targets

cols = ['nagi', 'agi']
xcols = ['ones', 'c00100']
targets = irstot[cols].iloc[3]
targets.agi = targets.agi * 1000.
targets = np.asarray(targets, dtype=float)
type(targets)
targets

wh = np.asarray(stub.s006)
type(wh)

# xmat = stub[['c00100']]
# xmat = np.array()
xmat = np.asarray(stub[xcols], dtype=float)
xmat.shape

x0 = np.ones(wh.size)

t0 = constraints(x0, wh, xmat)
pdiff0 = t0 / targets * 100 - 100
pdiff0
comp[['npdiff', 'wpdiff']].iloc[2]

rwp = rw.Reweight(wh, xmat, targets)
x, info = rwp.reweight(xlb=0.1, xub=10,
                       crange=.0001,
                       ccgoal=10, objgoal=100,
                       max_iter=50)
info['status_msg']

np.quantile(x, [0, .1, .25, .5, .75, .9, 1])

t1 = constraints(x, wh, xmat)
pdiff1 = t1 / targets * 100 - 100
pdiff1

data = load_wine()
wine = pd.DataFrame(data.data,
                    columns=data.feature_names)
wine.head()


# %% misc

df2.pivot_table(index='IRS_STUB',
               margins=True,
               margins_name='0',  # defaults to 'All'
               aggfunc=sum)

# map puf names to irstot variable values

df.info()
desc = df.describe()
cols = df.columns

tmp = irstot.loc[(irstot.src == '18in11si.xls') &
             (irstot.irsstub == 0) &
             (irstot.variable == 'nret_all')]
nret_all = tmp.iloc[0].value

df.s006.sum() / nret_all * 100 - 100  # -+1.4%
# djb pick up here ----

retcount = df.loc[df['c00100'] >= 10e6].s006.sum()
rec = irstot.loc[(irstot.src == '18in11si.xls') &
                 (irstot.irsstub == 19) &
                 (irstot.variable == 'nret_all')]
irscount = rec.iloc[0].value
retcount / irscount * 100 - 100  # +7%

df["IRS_STUB"] = pd.cut(
    df["c00100"],
    IRS_AGI_STUBS,
    labels=list(range(1, len(IRS_AGI_STUBS))),
    right=False,
)

df['wagi'] = df['s006'] * df['c00100']
grouped = df.groupby('IRS_STUB')

comp = irstot.drop(0)[['incrange', 'nagi', 'agi']]
comp['nagi'] = comp['nagi'].astype(float)
comp['nsums'] = grouped.s006.sum()
comp['ndiff'] = comp['nsums'] - comp['nagi']
comp['npdiff'] = comp['ndiff'] / comp['nagi'] * 100
comp['wagi'] = grouped.wagi.sum() / 1000
comp['wdiff'] = comp['wagi'] - comp['agi']
comp['wpdiff'] = comp['wdiff'] / comp['agi'] * 100
comp['wdiff_pctagi'] = comp.wdiff / sum(comp.agi) * 100
comp
comp.round(1)


totals = comp.drop(columns=['incrange', 'npdiff', 'wpdiff', 'wdiff_pctagi']).sum()
totals.ndiff / totals.nagi * 100  # 1.4%
totals.wdiff / totals.agi * 100  # 3.1%


# %% reweight the 2018 puf
# pick an income range to reweight and hit the number of returns and the amount of AGI

def constraints(x, wh, xmat):
    return np.dot(x * wh, xmat)

stub = df.loc[df['IRS_STUB'] == 3].copy()
stub['ones'] = 1.0

cols = ['nagi', 'agi']
xcols = ['ones', 'c00100']
targets = irstot[cols].iloc[3]
targets.agi = targets.agi * 1000.
targets = np.asarray(targets, dtype=float)
type(targets)
targets

wh = np.asarray(stub.s006)
type(wh)

# xmat = stub[['c00100']]
# xmat = np.array()
xmat = np.asarray(stub[xcols], dtype=float)
xmat.shape

x0 = np.ones(wh.size)

t0 = constraints(x0, wh, xmat)
pdiff0 = t0 / targets * 100 - 100
pdiff0
comp[['npdiff', 'wpdiff']].iloc[2]

rwp = rw.Reweight(wh, xmat, targets)
x, info = rwp.reweight(xlb=0.1, xub=10,
                       crange=.0001,
                       ccgoal=10, objgoal=100,
                       max_iter=50)
info['status_msg']

np.quantile(x, [0, .1, .25, .5, .75, .9, 1])

t1 = constraints(x, wh, xmat)
pdiff1 = t1 / targets * 100 - 100
pdiff1


# %% notes
# Peter's mappings of puf to historical table 2
# "n1": "N1",  # Total population
# "mars1_n": "MARS1",  # Single returns number
# "mars2_n": "MARS2",  # Joint returns number
# "c00100": "A00100",  # AGI amount
# "e00200": "A00200",  # Salary and wage amount
# "e00200_n": "N00200",  # Salary and wage number
# "c01000": "A01000",  # Capital gains amount
# "c01000_n": "N01000",  # Capital gains number
# "c04470": "A04470",  # Itemized deduction amount (0 if standard deduction)
# "c04470_n": "N04470",  # Itemized deduction number (0 if standard deduction)
# "c17000": "A17000",  # Medical expenses deducted amount
# "c17000_n": "N17000",  # Medical expenses deducted number
# "c04800": "A04800",  # Taxable income amount
# "c04800_n": "N04800",  # Taxable income number
# "c05800": "A05800",  # Regular tax before credits amount
# "c05800_n": "N05800",  # Regular tax before credits amount
# "c09600": "A09600",  # AMT amount
# "c09600_n": "N09600",  # AMT number
# "e00700": "A00700",  # SALT amount
# "e00700_n": "N00700",  # SALT number

    # Maps PUF variable names to HT2 variable names
VAR_CROSSWALK = {
    "n1": "N1",  # Total population
    "mars1_n": "MARS1",  # Single returns number
    "mars2_n": "MARS2",  # Joint returns number
    "c00100": "A00100",  # AGI amount
    "e00200": "A00200",  # Salary and wage amount
    "e00200_n": "N00200",  # Salary and wage number
    "c01000": "A01000",  # Capital gains amount
    "c01000_n": "N01000",  # Capital gains number
    "c04470": "A04470",  # Itemized deduction amount (0 if standard deduction)
    "c04470_n": "N04470",  # Itemized deduction number (0 if standard deduction)
    "c17000": "A17000",  # Medical expenses deducted amount
    "c17000_n": "N17000",  # Medical expenses deducted number
    "c04800": "A04800",  # Taxable income amount
    "c04800_n": "N04800",  # Taxable income number
    "c05800": "A05800",  # Regular tax before credits amount
    "c05800_n": "N05800",  # Regular tax before credits amount
    "c09600": "A09600",  # AMT amount
    "c09600_n": "N09600",  # AMT number
    "e00700": "A00700",  # SALT amount
    "e00700_n": "N00700",  # SALT number
}

