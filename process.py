import numpy as np
import xarray as xr
import os
import math
import matplotlib.pyplot as plt

# We import a model trained with ARTMO. Vegetation trait specific, this one is specified for 21 bands (Sentinel 3)
from model import FAPAR_model

# We open a satellite image (Sentinel-3 with 21 bands)
S3_dir = r"data/S3_scene.nc"
S3_dataset = xr.open_dataset(S3_dir)
S3_array = S3_dataset.to_array().values


def GPR_mapping(satellite_scene,
                hyp_ell_GREEN,
                mx_GREEN,sx_GREEN, X_train_GREEN, mean_model_GREEN,
                hyp_sig_GREEN, XDX_pre_calc_GREEN, alpha_coefficients_GREEN,
                Linv_pre_calc_GREEN):

    
    bands, ydim,xdim = satellite_scene.shape

    variable_map = np.empty((ydim,xdim))
    uncertainty_map = np.empty((ydim,xdim))
    combined = np.empty((2,ydim,xdim))

    for f in range(0,ydim):
        for v in range(0,xdim):

            pixel_spectra = satellite_scene[:,f,v]

            im_norm_ell2D_hypell  = ((pixel_spectra - mx_GREEN) / sx_GREEN) * hyp_ell_GREEN
            im_norm_ell2D_hypell = im_norm_ell2D_hypell.reshape(-1, 1)

            im_norm_ell2D  = ((pixel_spectra - mx_GREEN) / sx_GREEN)
            im_norm_ell2D = im_norm_ell2D.reshape(-1, 1)

            PtTPt = np.matmul(np.transpose(im_norm_ell2D_hypell) , im_norm_ell2D ).ravel() * (-0.5)
            PtTDX = np.matmul(X_train_GREEN,im_norm_ell2D_hypell).ravel().flatten()

            arg1 = np.exp(PtTPt) * hyp_sig_GREEN
            k_star_im = np.exp(PtTDX - (XDX_pre_calc_GREEN.ravel() * (0.5)))
            k_star = np.expand_dims(k_star_im, axis=0)

            mean_pred = (np.dot(k_star.ravel(),alpha_coefficients_GREEN.ravel()) * arg1) + mean_model_GREEN
            filterDown = np.greater(mean_pred,0).astype(int)
            mean_pred = mean_pred * filterDown

            k_star_uncert_im = np.exp(PtTDX - (XDX_pre_calc_GREEN.ravel() * (0.5))) * arg1

            k_star_uncert = np.expand_dims(k_star_uncert_im, axis=0)

            Vvector =  np.matmul(Linv_pre_calc_GREEN, k_star_uncert.reshape(-1, 1)).ravel()
 
            diff = np.expand_dims(hyp_sig_GREEN, axis=0) - np.dot(Vvector, Vvector)
            Variance = math.sqrt(abs(diff).item())


            combined[0, f, v] = mean_pred.item() 
            combined[1, f, v] = Variance

    return combined

# We call the Gaussian Process Regression function
# obtaining a map with 2 bands: mean estimate and uncertainty
mean_estimate, uncertainty_map = GPR_mapping(S3_array,       
                             FAPAR_model.hyp_ell_GREEN,
                             FAPAR_model.mx_GREEN,
                             FAPAR_model.sx_GREEN,
                             FAPAR_model.X_train_GREEN,
                             FAPAR_model.mean_model_GREEN,
                             FAPAR_model.hyp_sig_GREEN,
                             FAPAR_model.XDX_pre_calc_GREEN,
                             FAPAR_model.alpha_coefficients_GREEN,
                             FAPAR_model.Linv_pre_calc_GREEN)



plt.imshow(mean_estimate, vmin=0, vmax=1)
plt.title("Mean Estimate of FAPAR")
plt.colorbar()
plt.show()

plt.imshow(uncertainty_map, vmin=0, vmax=1)
plt.title("Uncertainty Map")
plt.colorbar()
plt.show()


