import numpy as np
import sys
import nibabel as nib
import pandas as pd
import os

# check for correct number of arguments
if len(sys.argv) == 1:
    print("Usage: python Rest_fc.py <job_n> <atlas> <subj_IDs> <HCP_dir> <out_dir>")
    print("\tjob_n: The job number for indexing subjects")
    print("\tatlas: The atlas to apply to the npy dtseries")
    print("\tsubj_IDs: A .txt file containing the list of subject IDs separated by newlines")
    print("\tHCP_dir: The directory containing the HCP data (with HCP directory structure)")
    print("\tout_dir: The output directory for the SVD embeddings")
    sys.exit(1)
elif len(sys.argv) < 6:
    print("Error: Missing arguments")
    sys.exit(1)

# read arguments
job_n=int(sys.argv[1])
atlas=sys.argv[2]
subj_IDs=sys.argv[3]
hcp_dir=sys.argv[4]
out_dir=sys.argv[5]
subj_IDs=np.loadtxt(subj_IDs).astype("int32")
subj=subj_IDs[job_n-1]
print("\n", "-"*20, f"\n{subj}")


# cortex = np.hstack([hcp.vertex_info.grayl, hcp.vertex_info.grayr + hcp.vertex_info.num_meshl])
labels = nib.load(atlas).get_fdata()[0].astype("int32")
concat_tseries = np.load(f'{out_dir}/{subj}.rfMRI_REST_All_Atlas_MSMAll_hp2000_smooth.npy')

concat_tseries = pd.DataFrame(concat_tseries)
concat_tseries["lab"] = labels
nw_tseries = concat_tseries.groupby("lab").agg("mean").drop(0)
fc = nw_tseries.T.corr(method="pearson")

fc.to_csv(f"{out_dir}/{subj}.REST_All_fcMatrix.csv", index=False, header=False)
os.remove(f'{out_dir}/{subj}.rfMRI_REST_All_Atlas_MSMAll_hp2000_smooth.npy')