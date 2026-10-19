#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -P = total number of processors
# TESTRUN:
# python3 -u draw_joint_samples.py -t cluster.toml -P 1 -p 0 -s 1000 -d test4
# REAL RUN:
# python3 -u draw_joint_samples.py -t cluster.toml -P ? -p ? -s 10000000 -d clustered
# (here, P and p come from draw_joint_samples.sh)

# https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html

##############################################################################

from math import log as ln, log10 as lg, sqrt, ceil
import os, sys
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
# https://matplotlib.org/basemap/stable/users/examples.html
# from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
# https://xarray-test.readthedocs.io/en/latest/index.html
# https://tutorial.xarray.dev/workshops/online-tutorial-series/01_xarray_fundamentals.html
# https://docs.xarray.dev/en/stable/user-guide/io.html
# https://docs.xarray.dev/en/stable/user-guide/reshaping.html
import xarray as xr
import hdbscan
from sklearn.preprocessing import StandardScaler
from math import log, log10, exp
#qqq from icecream import ic as qqq ; qqq.configureOutput(prefix='qqq ', includeContext=True)
from textwrap import fill
from datetime import datetime
from copy import deepcopy

# For plot_emac:
from matplotlib.backends.backend_pdf import PdfPages

from collections import Counter

from sklearn.preprocessing import robust_scale, scale
from random import random, randint, choice, gauss

from shared import get_var

##############################################################################

HLINE =  '-' * 78

TESTRUN = False # qqq True
ANALYZE = True
VERBOSE = False

SPECIES = ['O3', 'OH', 'NO', 'NO2', 'C5H8', 'BENZENE', 'CO']

COLORMAP = 'gist_ncar'
#COLORMAP = 'prism'
PROJ = ccrs.PlateCarree()

LOGSCALE = True

# Initial parameters:
PARAMS_0 = {'min_cluster_size':          30,
            'min_samples':               22,
            'cluster_selection_epsilon': 0.42,
            'metric':                    'manhattan',
            'alpha':                     1.0,
            'cluster_selection_method':  'leaf'}

xr.set_options(display_expand_attrs=False, display_expand_data=False)

##############################################################################

def ds_info(ds):
    infostring  = f'Vars:        {list(ds)}'
    infostring += f'\nSizes:       {dict(ds.sizes)}'
    infostring += f'\n{ds.coords}'
    #return infostring
    return f'\n{HLINE}\n{infostring}\n{HLINE}'

##############################################################################

def print_func_name():
    # https://stackoverflow.com/questions/5067604/determine-function-name-from-within-that-function#comment118968103_17366561
    #sys._getframe().f_code.co_name
    print(f'\n{HLINE}\n{sys._getframe().f_back.f_code.co_name}\n{HLINE}')

##############################################################################

def get_emac_data(ncfile, species, timestep=0, level=0):

    print_func_name()

    species_with_suffix = [f'{spec0}_start' for spec0 in species]

    # Load netcdf file into an xarray dataset:
    ds = xr.open_dataset(ncfile)

    # Select timestep and level:
    ds = ds.isel(time=timestep, eta=level)
    print(f'\nds (timestep={timestep}, level={level}):', ds)

    # Create list of all variables in dataset:
    all_var = list(ds.data_vars)
    # Create list of all variables on lat,lon grid:
    all_species = [var for var in all_var if ds[var].dims==('lat', 'lon')]
    print('\n' + fill(f'All species: ' + ', '.join(all_species), subsequent_indent='  '))
    print(fill(f'Selected species: ' + ', '.join(species), subsequent_indent='  '))

    # Return dataset which contains only the selected species:
    return ds[species_with_suffix]

##############################################################################

def plot_emac(ds):

    print_func_name() ; print()

    # https://mom6-analysiscookbook.readthedocs.io/en/latest/notebooks/Plotting.html
    # https://medium.com/@lubomirfranko/climate-data-visualisation-with-python-visualise-climate-data-using-cartopy-and-xarray-cf35a60ca8ee
    pdf_pages = PdfPages('emac.pdf') # open pdf
    for spec in ds.keys():
        val = ds[spec].values
        vmin = val[val>1E-20].min() # ignore values<1E-20 mol/mol
        vmax = val.max()
        print(f'Plotting {spec:10} (MIN={vmin:10.3E}, MAX={vmax:10.3E})')
        fig, axs = plt.subplots(2, 1, subplot_kw={'projection':PROJ})
        # --------------------------------------------------------------------
        # Linear plot:
        ax = axs[0]
        fig = ds[spec].plot(norm=colors.Normalize(), vmin=vmin, vmax=vmax,
                            cmap=COLORMAP, ax=ax)
        fig.axes.coastlines()
        ax.set_title(f'{spec} (LIN)')
        # --------------------------------------------------------------------
        # Logarithmic plot:
        ax = axs[1]
        fig = ds[spec].plot(norm=colors.LogNorm(), vmin=vmin, vmax=vmax,
                            cmap=COLORMAP, ax=ax)
        fig.axes.coastlines()
        ax.set_title(f'{spec} (LOG)')
        # --------------------------------------------------------------------
        plt.tight_layout()
        pdf_pages.savefig() # save current page and create a new page in pdf
    pdf_pages.close() # close pdf

##############################################################################

def cluster_data_3d(ds_allvar, variables, samples_per_file, fname):

    if VERBOSE: print_func_name() ; print()

    subsets = []

    da, units = get_var(ds_allvar, 'lat')
    ds_allvar['lat_degN'] = da # latitude as a normal variable (not a dimension variable)
    da, units = get_var(ds_allvar, 'lon')
    ds_allvar['lon_degE'] = da # longitude as a normal variable (not a dimension variable)
    da, units = get_var(ds_allvar, 'p')
    da.attrs['long_name'] = 'pressure'
    ds_allvar['p'] = da
    # ds_allvar['ilon'] = [ilon+1 for ilon in range(ds_allvar.sizes['lon'])]
    # ds_allvar['ilat'] = [ilat+1 for ilat in range(ds_allvar.sizes['lat'])]    
    #variables.extend(['lat_degN', 'lon_degE', 'p', 'ilon', 'ilat'])
    variables.extend(['lat_degN', 'lon_degE', 'p'])
    ds = ds_allvar[variables]
    # remove suffix "_start" from variable names:
    rename_dict = {}
    for varname in ds.data_vars:
        if (varname.endswith('_start')):
            rename_dict[varname] = varname.removesuffix('_start')
    ds = ds.rename_vars(rename_dict)

    N_times   = ds.sizes['time']
    N_lev     = ds.sizes['eta']
    if TESTRUN:
        levellist = [i for i in range(N_lev) if i%11==8] # use only a few levels for testing
        timelist = [3, 5] # use only a few time steps for testing
        N_samples = int(0.02*ds.sizes['lat']*ds.sizes['lon']) # simply use 2%
        print(f'N_samples = {N_samples}')
    else:
        levellist = range(N_lev)
        timelist  = range(N_times)
        N_samples = ceil(samples_per_file/(N_times*N_lev))
        print(f'Taking >={samples_per_file} samples from current file ' +
              f'({N_times} times * {N_lev} levels * {N_samples} samples = ' +
              f'{N_times*len(levellist)*N_samples})')

    # Open pdf:
    fname_stem = os.path.splitext(os.path.basename(fname))[0]
    with PdfPages(f'clusters-{fname_stem}.pdf') as pdf_pages:
        for t in timelist:
            for ilev in levellist:
                timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
                print(f'{timestamp}: time={t}, level={ilev}')
                ds_subset = ds.isel(time=t,eta=ilev)
                timestring = ds_subset['time'].dt.strftime('%Y-%m-%d-%H:%M:%S').values
                # Stack lon and lat into 1 dimension:
                # https://xarray-test.readthedocs.io/en/latest/reshaping.html#stack-and-unstack
                if VERBOSE: print('\nds_subset (lon+lat stacked):')
                ds_subset_flat = ds_subset.stack(location=('lat', 'lon'))
                if VERBOSE: print(ds_info(ds_subset_flat))
                # Prepare data:
                # Make a deep copy to avoid that ds_subset_flat is changed:
                ds_subset_flat_SPECIES = deepcopy(ds_subset_flat[SPECIES])
                da = prepare_data_for_clustering(ds_subset_flat_SPECIES)
                if VERBOSE: print(ds_info(da))
                # Cluster the data and produce random indices, weighted by cluster sizes:
                clstrsizes_1d, selected_1d, selected_idx, infodict = cluster_data(
                    da, N_samples)
                ds_selected = ds_subset_flat.isel(location=selected_idx)
                if VERBOSE: print('ds_selected:', ds_info(ds_selected))
                ds_selected2 = ds_selected.drop_vars(['location', 'lon', 'lat'])
                if VERBOSE: print('ds_selected2:', ds_info(ds_selected2))
                ds_selected2['ilev'] = ilev
                subsets.append(ds_selected2)
                #qqq if (ANALYZE):
                #qqq if (ANALYZE and random()>0.9): # make plots for 10 % of the data
                if (ANALYZE and random()>0.999): # make plots for 0.1 % of the data
                    info = (f'{fname_stem},  t={t} ({timestring}),  level={ilev+1},  ' +
                            f'N_clstrs={infodict["N_clstrs"]} ' +
                            f'(biggest={infodict["biggest_clstr"]},  ' +
                            f'unclstred={infodict["N_unclstred"]})')
                    timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
                    print(f'{timestamp}: Start plot_clusters')
                    plot_clusters(ds_subset, clstrsizes_1d, selected_1d, pdf_pages, info)
                    timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
                    print(f'{timestamp}: End   plot_clusters')

    print('Concat datasets for all levels and times.\n')
    ds_flat_cleaned = xr.concat(subsets, dim='location', compat='equals',
                                coords='different', data_vars='all')
    if VERBOSE: print('ds_flat_cleaned:', ds_info(ds_flat_cleaned)) ; print(ds_flat_cleaned)
    if VERBOSE: print(ds_flat_cleaned['location'])

    return ds_flat_cleaned.rename({'location': 'sample_nr'})

##############################################################################

def prepare_data_for_clustering(ds_flat):

    if VERBOSE:
        print_func_name()
        print()
        if LOGSCALE:
            print(f'\n{HLINE}\nLOG SCALE\n{HLINE}\n')
        else:
            print(f'\n{HLINE}\nLIN SCALE\n{HLINE}\n')

    # Normalize values with scaler from sklearn:
    if VERBOSE: print('\nds_flat (normalized):')
    for spec in ds_flat.keys():
        # Choose lin/log type and corresponding best scaler:
        if LOGSCALE:
            ds_flat[spec].values = np.log(np.maximum(1E-30,ds_flat[spec].values))
            ds_flat[spec].values = robust_scale(ds_flat[spec].values)
            #ds_flat[spec].values = scale(ds_flat[spec].values)
        else:
            ds_flat[spec].values = robust_scale(ds_flat[spec].values)
    if VERBOSE: print(ds_flat)

    # Convert xarray dataset to xarray dataarray:
    if VERBOSE: print('\nda (dataarray):')
    da = ds_flat.to_dataarray()
    if VERBOSE: print(da)

    # Swap species and location:
    # https://xarray-test.readthedocs.io/en/latest/reshaping.html#reordering-dimensions
    if VERBOSE: print('\nda (species and location swapped):')
    da = da.transpose('location', 'variable')
    if VERBOSE: print(da)

    return da

##############################################################################

def get_params():

    params = dict()

    # min_cluster_size: Minimum number of points required to form a
    # cluster (default: 5). Smaller values create more clusters but may
    # include noise. Larger values create fewer, more robust clusters. A
    # good starting point is 1-2% of your dataset size.
    params['min_cluster_size'] = 1 + int(50. * exp(gauss(0.,1.))) if random()>0.3 else 50
    #params['min_cluster_size'] = PARAMS_0['min_cluster_size']

    # min_samples: Number of samples in a neighborhood for a point to be
    # considered a core point (default: None, uses min_cluster_size).
    # Controls the local density requirement. Should typically be set to
    # about half of min_cluster_size.
    #params['min_samples'] = 1 + int(0.5 * min_cluster_size*exp(gauss(0.,0.2))) if random()>0.3 else 20
    params['min_samples'] = int(PARAMS_0['min_samples'])# * exp(gauss(0.,0.02))) + 1
    #params['min_samples'] = PARAMS_0['min_samples']

    # cluster_selection_epsilon: Distance threshold for cluster
    # selection (default: 0.0). Points within this distance are
    # considered for cluster membership. Lower values create more
    # conservative clustering. Typically left at 0.0 unless you need to
    # merge nearby clusters.
    params['cluster_selection_epsilon'] = (PARAMS_0['cluster_selection_epsilon']
                                           * exp(gauss(0.,0.05)))
    #params['cluster_selection_epsilon'] = PARAMS_0['cluster_selection_epsilon']

    # metric: For distance calculations (default: 'euclidean')
    # params['metric'] = choice(['braycurtis', 'chebyshev', 'cityblock', 'euclidean',
    #                  'infinity', 'l1', 'l2', 'manhattan', 'p'])
    params['metric'] = choice(['cityblock', 'l1', 'manhattan'])
    #params['metric'] = 'manhattan'

    # alpha: Distance scaling parameter for cluster selection (default:
    # 1.0). Higher values create more conservative clustering. Typically
    # left at 1.0 unless you need to adjust cluster selection behavior.
    params['alpha'] = 1.0

    # cluster_selection_method: determine how it selects flat clusters
    # from the cluster tree hierarchy. Default = 'eom' for Excess of
    # Mass. For small homogeneous clusters, try 'leaf', producing many
    # small homogeneous clusters.
    params['cluster_selection_method'] = choice(['eom', 'leaf'])
    #params['cluster_selection_method'] = PARAMS_0['cluster_selection_method']

    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
    # https://hdbscan.readthedocs.io/en/latest/parameter_selection.html
    # https://mbrenndoerfer.com/writing/hdbscan-hierarchical-density-based-clustering-automatic-cluster-selection

    return params

##############################################################################

def cluster_data(da, N_samples, params=PARAMS_0):

    if VERBOSE: print_func_name() ; print()

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size          = params['min_cluster_size'],
        min_samples               = params['min_samples'],
        cluster_selection_epsilon = params['cluster_selection_epsilon'],
        # max_cluster_size        = None,
        metric                    = params['metric'],
        # metric_params           = None,
        alpha                     = params['alpha'],
        # algorithm               = 'auto',
        # leaf_size               = 40,
        # n_jobs                  = None,
        cluster_selection_method  = params['cluster_selection_method'],
        # allow_single_cluster    = False,
        # store_centers           = None,
        # copy                    = 'warn',
        )

    clusterer.fit(da)

    if VERBOSE: print('\nclstrsizes_1d (numpy array):')
    clstrsizes_1d = clusterer.labels_
    if VERBOSE: print('Shape: ', clstrsizes_1d.shape)
    clstrsizes = Counter(clstrsizes_1d)
    if VERBOSE: [print(f'{key:3}: {value}') for key, value in sorted(clstrsizes.items())]

    # N_lat*N_lon=ds.sizes['lat']*ds.sizes['lon']=96*192=18432=clusterer.labels_.size:
    N_location = da.sizes['location']
    # Define weights as a function of cluster size (default=1 for unclustered boxes):
    weights_dict = dict()
    for key, value in clstrsizes.items():
        weights_dict[key] = 1. if (key==-1) else 1./(ln(value)+1.)
    # Define weights for each box:
    weights = np.zeros(N_location, dtype=float) # 96*192 = 18432
    for key, value in weights_dict.items():
        weights[clstrsizes_1d==key] = value
    weights_sum = weights.sum()
    # Normalized weights in numpy array (for each box):
    weights /= weights_sum
    # Normalized weights as a dict (for each cluster):
    norm_weigths = {key: weights_dict[key]/weights_sum for key in weights_dict.keys()}
    # Select some boxes:
    selected_idx = np.random.choice(N_location, N_samples, p=weights, replace=False)
    if VERBOSE: print('\nIndices of selected boxes:', selected_idx)

    selected_1d = np.full((N_location), False)
    for idx in selected_idx:
        selected_1d[idx] = True

    if (ANALYZE):
        analyze_clustering_and_selection(
            N_location, params, N_samples, weights_dict, norm_weigths,
            clstrsizes, clstrsizes_1d, selected_1d)

    infodict = {}
    infodict['loginfo'] = 'LOG' if LOGSCALE else 'LIN'
    infodict['params'] = params
    infodict['N_clstrs'] = clstrsizes_1d.max() + 1
    infodict['biggest_clstr'] = max([clstrsizes[key] for key in clstrsizes if key != -1])
    infodict['N_unclstred'] = clstrsizes[-1]

    return clstrsizes_1d, selected_1d, selected_idx, infodict

##############################################################################

def analyze_clustering_and_selection(
        N_location, params, N_samples, weights_dict, norm_weigths,
        clstrsizes, clstrsizes_1d, selected_1d):
    n_slctd = Counter(clstrsizes_1d[selected_1d])
    print('key  weights_dict norm_weigths clstrsizes slctd expctd slctd/clstr '
          + 'slctd/expctd')
    for key, weight in sorted(weights_dict.items()):
        norm_weigth = norm_weigths[key]
        clstrsize   = clstrsizes[key]
        slctd       = n_slctd[key]
        print(f'{key:3}: {weight:12.5f} {norm_weigth:12.4E} {clstrsize:10} '
              + f'{slctd:5} {N_samples*norm_weigth*clstrsize:6.2f} ' # slctd, expctd
              + f'{100.*slctd/clstrsize:9.2f} % ' # slctd/clstr
              + f'{slctd/(N_samples*norm_weigth*clstrsize):12.5f}') # slctd/expctd
    print(f'all:   unclstrd=1        sum=1      {N_location}  {N_samples:4}   '
          + f'{N_samples:4} {100.*N_samples/(N_location):9.2f} %     approx 1\n')

##############################################################################

def plot_clusters(ds, clstrsizes_1d, selected_1d, pdf_pages, info):

    N_lat = ds.sizes['lat'] #  96
    N_lon = ds.sizes['lon'] # 192
    N_clstrs = clstrsizes_1d.max() + 1
    selected_2d = np.reshape(selected_1d, (N_lat, N_lon))
    if VERBOSE: print('\nclstrsizes_2d (numpy array, lon+lat separated again):')
    clstrsizes_2d = np.reshape(clstrsizes_1d, (N_lat, N_lon))

    # Add clusters and selected as new dataarrays to dataset ds:
    ds['clusters'] = (('lat','lon'), clstrsizes_2d)
    ds['selected'] = (('lat','lon'), selected_2d)
            
    plt.rcParams.update({'figure.max_open_warning': 0})
    # https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar
    PLOTROWS = 3
    PLOTCOLS = 3
    fig, axs = plt.subplots(PLOTROWS, PLOTCOLS, figsize=(13,7),
                            subplot_kw={'projection':PROJ})
    fig.suptitle(info)
    # ---
    ax = axs[0,0]
    colorbar = {'label':'', 'ticks':np.arange(-1,N_clstrs), 'shrink':0.7, 'pad':0.05}
                #'orientation':'horizontal',
    fig = ds['clusters'].plot(levels=N_clstrs+2,
                              cmap=plt.get_cmap('nipy_spectral'),
                              transform=PROJ, cbar_kwargs=colorbar,
                              vmin=-1-0.5, vmax=N_clstrs-0.5,
                              ax=ax)
    fig.colorbar.ax.tick_params(labelsize=7)
    fig.axes.coastlines(color='white')
    ax.set_title('clusters')
    # ---
    ax = axs[0,1]
    colorbar = {'label':'', 'ticks':[0,1], 'shrink':0.7, 'pad':0.05}
    fig = ds['selected'].plot(levels=3, cmap=COLORMAP,
                              transform=PROJ, cbar_kwargs=colorbar,
                              vmin=-0.5, vmax=1.5,
                              ax=ax)
    fig.colorbar.ax.tick_params(labelsize=7)
    fig.axes.coastlines()
    N_slctd = Counter(selected_1d)[True]
    percent_selected = (100.*N_slctd)/selected_1d.size
    ax.set_title(f'{N_slctd} selected ({percent_selected:.2f} %)')
    # ---
    for i, spec in enumerate(SPECIES):
        i_subplot = i + 2 # 2 subplots have already been made above
        row = i_subplot//PLOTCOLS
        col = i_subplot%PLOTCOLS
        ax = axs[row, col]
        val = ds[spec].values
        vmin = val[val>1E-20].min() # ignore values<1E-20 mol/mol
        vmax = val.max()
        #print(f'Plotting {spec:10} (MIN={vmin:10.3E}, MAX={vmax:10.3E})')
        colorbar = {'label':'', 'shrink':0.7, 'pad':.05}
        fig = ds[spec].plot(norm=colors.LogNorm(), vmin=vmin, vmax=vmax,
                            cmap=COLORMAP, cbar_kwargs=colorbar, ax=ax)
        fig.colorbar.ax.tick_params(labelsize=7, which='both')
        plt.title('')
        fig.axes.coastlines()
        ax.set_title(f'{spec} (LOG)')
    # ---
    plt.tight_layout()
    pdf_pages.savefig() # save current page and create a new page in pdf

##############################################################################

def plot_scatter(ds):

    print_func_name() ; print()

    pdf_pages = PdfPages('scatter.pdf') # open pdf
    plt.rcParams.update({'figure.max_open_warning': 0})
    for spec1 in SPECIES:
        for spec2 in SPECIES:
            if (not(spec1==spec2)):
                print(f'Plotting {spec1} vs {spec2}...')
                plt.figure()
                sc = plt.scatter(ds[spec1], ds[spec2],
                                 c=ds['clusters'], s=10, marker='.',
                                 linewidths=0, cmap=COLORMAP)
                plt.xlabel(spec1)
                plt.ylabel(spec2)
                plt.colorbar(sc)
                pdf_pages.savefig() # save current page and create a new page in pdf
    pdf_pages.close() # close pdf

##############################################################################

if __name__ == '__main__':

    TIMESTEP = 0
    N_runs = 3
    N_samples = 200

    # Read data from EMAC netcdf file:
    #ncfile = '../input/emac.nc'
    ncfile = '../input/MOM-ML4GPK.1r__20100131_1300_mecca_gp.nc'
    ds = get_emac_data(ncfile, SPECIES, timestep=TIMESTEP)

    # Plot EMAC data:
    #qqq plot_emac(ds)

    # Stack lon and lat into 1 dimension:
    # https://xarray-test.readthedocs.io/en/latest/reshaping.html#stack-and-unstack
    if VERBOSE: print('\nds (lon+lat stacked):')
    ds_flat = ds.stack(location=('lat', 'lon'))
    if VERBOSE: print(ds_flat)

    # Prepare data:
    da = prepare_data_for_clustering(ds_flat)

    pdf_pages = PdfPages('clusters.pdf') # open pdf
    for i in range(N_runs):
        # Perform clustering:
        clstrsizes_1d, selected_1d, selected_idx, infodict = cluster_data(
            da, N_samples, params=get_params())
        params = infodict['params']
        info = (f'{i+1:4}: {infodict["loginfo"]}, ' +
                f'{params["min_cluster_size"]:>4}, {params["min_samples"]:>4}, ' +
                f'{params["cluster_selection_epsilon"]:4.2f}, {params["metric"]:>10}, ' +
                f'{params["alpha"]:5.2f}, {params["cluster_selection_method"]:>4} -> ' +
                f'{infodict["N_clstrs"]:>5}, {infodict["biggest_clstr"]:>5}, ' +
                f'{infodict["N_unclstred"]:>5}')
        # Plot clusters:
        plot_clusters(ds, clstrsizes_1d, selected_1d, pdf_pages, info)
        # Make scatter plots:
        #qqq plot_scatter(ds)
    pdf_pages.close() # close pdf

    print(f'\n{HLINE}\n')

##############################################################################
