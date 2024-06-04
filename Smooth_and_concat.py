import nibabel as nib
import numpy as np
import hcp_utils as hcp
import os, sys, subprocess as sp
import re

# check for correct number of arguments
if len(sys.argv) == 1:
    print("Usage: python preproc.py <job_n> <kernel> <subj_IDs> <HCP_dir> <out_dir>")
    print("\tjob_n: The job number for indexing subjects")
    print("\tkernel: The kernel size for spatial smoothing of the timeseries")
    print("\tsubj_IDs: A .txt file containing the list of subject IDs separated by newlines")
    print("\tHCP_dir: The directory containing the HCP data (with HCP directory structure)")
    print("\tout_dir: The output directory for the preprocessed data")
    sys.exit(1)
elif len(sys.argv) < 6:
    print("Error: Missing arguments")
    sys.exit(1)



def fullTs(funcs):
    ### function takes a list of functional runs corresponding to a given subject 
    Ldat=[]
    Rdat=[]
    for cifti in funcs:
        img=nib.load(cifti).get_fdata()
        img=hcp.normalize(img)
        Ldat.append(img[:, hcp.struct.cortex_left].T)
        Rdat.append(img[:, hcp.struct.cortex_right].T)
    
    Lcort=np.hstack(Ldat)    
    Rcort=np.hstack(Rdat)    
    
    cortTs=np.vstack([Lcort,Rcort])
    
    return cortTs


# read arguments
job_n=int(sys.argv[1])
kernel=int(sys.argv[2])
subj_IDs=sys.argv[3]
hcp_dir=sys.argv[4]
out_dir=sys.argv[5]
subj_IDs=np.loadtxt(subj_IDs).astype("int32")
subj=subj_IDs[job_n-1]
subj_dir=f"{hcp_dir}/{subj}" 
print("\n", "-"*20, f"\n{subj}")

# Find dtseries files for resting state runs
regex = re.compile('rfMRI_REST[1,2]{1}_[L,R]{2}_Atlas_MSMAll_hp2000_clean.dtseries.nii')
rest_runs = [f"{root}/{file}" for root, _, files in os.walk(subj_dir) for file in files if regex.match(file)]
print("Runs:\n\t", "\n\t".join(rest_runs))


# Pathts to save smoothed files
rest_runs_smooth = []
for run in rest_runs:
    run_smooth = run.replace("clean", "smooth").split("/")[-1]
    run_smooth = f"{out_dir}/{subj}.{run_smooth}"
    rest_runs_smooth.append(run_smooth)


# Verify existence of smoothed concatenated timeseries
concat_tseries = f'{out_dir}/{subj}.rfMRI_REST_All_Atlas_MSMAll_hp2000_smooth.npy'
if not os.path.exists(concat_tseries):

    # Smooth timeseries
    for i, j in zip(rest_runs, rest_runs_smooth):
        sp.run(f'wb_command -cifti-smoothing {i} {kernel} {kernel} COLUMN {j} \
            -left-surface {subj_dir}/MNINonLinear/fsaverage_LR32k/{subj}.L.midthickness_MSMAll.32k_fs_LR.surf.gii \
            -right-surface {subj_dir}/MNINonLinear/fsaverage_LR32k/{subj}.R.midthickness_MSMAll.32k_fs_LR.surf.gii', 
            shell=True)
        print(j, "saved")

    #save the concatenated timeseries
    np.save(concat_tseries, fullTs(rest_runs_smooth))
    print("\nConcatenated timeseries saved:\n", concat_tseries)

    # Remove temporary smoothed files
    for i in rest_runs_smooth:
        if os.path.exists(i):
         os.remove(i)
else:
    print("Concatenated timeseries already exists:\n", concat_tseries)
