
import numpy as np
import os, sys, re, time
from scipy.stats import zscore
from scipy.sparse.linalg import svds
from scipy.linalg import svdvals

# check for correct number of arguments
if len(sys.argv) == 1:
    print("Usage: python Individual_SVD.py <job_n> <rank> <subj_IDs> <HCP_dir> <out_dir>")
    print("\tjob_n: The job number for indexing subjects")
    print("\trank: The rank for the singular value decomposition")
    print("\tsubj_IDs: A .txt file containing the list of subject IDs separated by newlines")
    print("\tHCP_dir: The directory containing the HCP data (with HCP directory structure)")
    print("\tout_dir: The output directory for the SVD embeddings")
    sys.exit(1)
elif len(sys.argv) < 6:
    print("Error: Missing arguments")
    sys.exit(1)

# read arguments
job_n=int(sys.argv[1])
rank=int(sys.argv[2])
subj_IDs=sys.argv[3]
hcp_dir=sys.argv[4]
out_dir=sys.argv[5]
subj_IDs=np.loadtxt(subj_IDs).astype("int32")
subj=subj_IDs[job_n-1]
print("\n", "-"*20, f"\n{subj}")



# Check if the SVD embeddings already exist
regex = re.compile(str(subj) + '.SVD_[U,S,V]{1}.rfMRI_REST_All.npy')
svd_files = [f"{root}/{file}" for root, _, files in os.walk(out_dir) for file in files if regex.match(file)]
if len(svd_files) != 0:
    print("Existing SVD output:\n\t", "\n\t".join(svd_files))


if len(svd_files) != 3:

    # load and center data
    concat_tseries = f'{out_dir}/{subj}.rfMRI_REST_All_Atlas_MSMAll_hp2000_smooth.npy'
    X = np.load(concat_tseries)
    X = zscore(X, axis=1, ddof=1)
    mu = np.mean(X, axis=0)
    X -= mu


    # Compute individual embeddings
    u, s, vt = svds(X, k=rank, which="LM", random_state=0)
    sorter = np.argsort(-s)
    v = vt.T
    v = v[:, sorter]
    ut = u.T
    u = ut.T[:, sorter]
    s = s[sorter]

    exp_var = svdvals(X)**2
    exp_var = exp_var/np.sum(exp_var)
    exp_var = exp_var[np.argsort(-exp_var)]
    
    print(f"\nSVD explained variance: {exp_var[:rank]*100}%\nTotal: {np.sum(exp_var[:rank])*100}%")
    # Save embeddings
    np.save(f"{out_dir}/{subj}.SVD_U.rfMRI_REST_All.npy", u)
    np.save(f"{out_dir}/{subj}.SVD_S.rfMRI_REST_All.npy", s)
    np.save(f"{out_dir}/{subj}.SVD_V.rfMRI_REST_All.npy", v)
    np.save(f"{out_dir}/{subj}.SVD_exp.rfMRI_REST_All.npy", exp_var)

    print(f"\nSVD embeddings computed for {subj}:", 
          f"\n\t{out_dir}/{subj}.SVD_U.rfMRI_REST_All.npy",
          f"\n\t{out_dir}/{subj}.SVD_S.rfMRI_REST_All.npy", 
          f"\n\t{out_dir}/{subj}.SVD_V.rfMRI_REST_All.npy", 
          f"\n\t{out_dir}/{subj}.SVD_exp.rfMRI_REST_All.npy")

    os.remove(f"{out_dir}/{subj}.rfMRI_REST_All_Atlas_MSMAll_hp2000_smooth.npy")