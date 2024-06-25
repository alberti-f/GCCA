import numpy as np
from scipy import stats, linalg
from scipy.sparse.linalg import svds
from scipy.linalg import svdvals
from sklearn.preprocessing import normalize
import os, sys, time


# check for correct number of arguments
if len(sys.argv) == 1:
    print("Usage: python Individual_SVD.py <job_n> <rank> <subj_IDs> <HCP_dir> <out_dir>")
    print("\tjob_n: The job number for indexing subjects")
    print("\trank: The rank for the cross-individuals singular value decomposition")
    print("\tsubj_IDs: A .txt file containing the list of subject IDs separated by newlines")
    print("\tHCP_dir: The directory containing the HCP data (with HCP directory structure)")
    print("\tout_dir: The output directory for the individual canonical projections")
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
subj_dir=f"{hcp_dir}/{subj}"
print("\n", "-"*20, f"\n{subj}")

if not os.path.exists(f"{out_dir}/group_SVD_V.npy"):
    print("Computing group-level SVD")
    # Load individual SVD embeddings
    svd_results = out_dir + "/{0}.SVD_{1}.rfMRI_REST_All.npy"
    Uall=[np.load(svd_results.format(subj, "U")) for subj in subj_IDs]
    Uall = np.hstack(Uall)

    # Cross-individual SVD
    _, _, VV = svds(Uall, k=rank)
    VV = np.flip(VV.T, axis=1)
    VV = VV[:, : rank]

    exp_var = svdvals(Uall)**2
    exp_var = exp_var/np.sum(exp_var)
    exp_var = exp_var[np.argsort(-exp_var)]
    exp_var = exp_var[:rank]

    np.save(f"{out_dir}/group_SVD_V.npy", VV)
    np.save(f"{out_dir}/group_SVD_exp.npy", exp_var)

    print("\nGroup SVD embeddings computed successfully\n\t", f"{out_dir}/group_SVD_V.npy")



# Load individual SVD embeddings
N = len(subj_IDs)
svd_results = out_dir + "/{0}.SVD_{1}.rfMRI_REST_All.npy"  
U = np.load(svd_results.format(subj, "U"))
S = np.load(svd_results.format(subj, "S"))
V = np.load(svd_results.format(subj, "V"))
VV = np.load(f"{out_dir}/group_SVD_V.npy")
X = np.load(f"{out_dir}/{subj}.rfMRI_REST_All_Atlas_MSMAll_hp2000_smooth.npy")



# Compute projection matrix
projection_mats = []
idx_start = (job_n-1) * rank
idx_end = idx_start + rank
VVi = normalize(VV[idx_start:idx_end, :], "l2", axis=0)
A = np.sqrt(N - 1) * V
A = A @ (linalg.solve(np.diag(S), VVi))
print("\nProjection matrix computed")


# Compute canonical projections
X = stats.zscore(X, axis=1, ddof=1)
mu = np.mean(X, axis=0)
X -= mu
Xfit = X @ A
np.save(f"{out_dir}/{subj}.GCCA.npy", Xfit.T)
print("Canonical projections saved")


# Remove temporary files
if os.path.exists(f"{out_dir}/{subj}.GCCA.npy"):
    if os.path.exists(f"{out_dir}/{subj}.rfMRI_REST_All_Atlas_MSMAll_hp2000_smooth.npy"):
        os.remove(f"{out_dir}/{subj}.rfMRI_REST_All_Atlas_MSMAll_hp2000_smooth.npy")
    os.remove(f"{out_dir}/{subj}.SVD_U.rfMRI_REST_All.npy")
    os.remove(f"{out_dir}/{subj}.SVD_S.rfMRI_REST_All.npy")
    os.remove(f"{out_dir}/{subj}.SVD_V.rfMRI_REST_All.npy")




# Compute SVD explained variance in group embedding
if job_n == N:
    computed_gcca = False

    while not all(computed_gcca):
        computed_gcca = [os.exists(f"{out_dir}/{id}.GCCA.npy") for id in subj_IDs]
        time.sleep(30)

    exp_subj = [np.load(f"{out_dir}/{id}.SVD_exp.rfMRI_REST_All.npy") for id in subj_IDs]
    exp_group = np.load(f"{out_dir}/group_SVD_exp.npy").sum()
    exp_subj = np.array(exp_subj) * exp_group

    np.save(f"{out_dir}/group_SVD_exp_subj.npy", exp_subj)
    
    for id in subj_IDs:
        if os.path.exists(f"{out_dir}/{id}.SVD_exp.rfMRI_REST_All.npy"):
            os.remove(f"{out_dir}/{id}.SVD_exp.rfMRI_REST_All.npy")


        