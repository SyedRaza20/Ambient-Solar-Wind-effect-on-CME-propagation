# author meta data:
__name__ = "Syed Raza"
__email__ = "sar0033@uah.edu"

# importing my modules
import sw_data_process as sw
import machine_learning_protocols as ml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Data analysis pipeline:
############################################################################################################
# get the file path for the donki data
file_path = "DONKI_new.xlsx"

# Call the clean donki function on this file path
donki = sw.clean_donki(file_path)

# Getting the cleaned OMNI data
omni = sw.clean_omni_data("OMNI_COHO1HR_MERGED_MAG_PLASMA_137596.txt")

# Getting the observed processed wind and the simulated processed wind data frames:
processed_wind_obs = sw.process_observed_parameters(omni, donki)
processed_wind_sim = sw.process_simulated_parameters(donki, "ENLIL_data")

# the cme data frame:
cme_data = sw.residual_wind(processed_wind_obs, processed_wind_sim, donki)

sw.plot_cme_transit(cme_data['cme_transit_obs'], 'Observed CME Transit', 'blue', 'observed_transit')
sw.plot_cme_transit(cme_data['cme_transit_sim'], 'ENLIL CME Transit', 'red', 'enlil_transit')


# making the parameter info for plotting the data analysis plots:
parameter_info = {
    "B_diff": ("$\Delta B_{SW}$ (G)", "figures/correlation_B.png"),
    "V_diff": ("$\Delta v_{SW}$ (km/s)", "figures/correlation_v.png"),
    "n_diff": ("$\Delta n_{SW}$ ($cm^{-3}$)", "figures/correlation_n.png"),
    "T_diff": ("$\Delta T_{SW}$ (K)", "figures/correlation_T.png"),
    "P_diff": ("$\Delta P_{SW}$ (Barye)", "figures/correlation_P.png"),
    "long": ("Longitude-CONE model (degrees)", "figures/correlation_long.png"),
    "lat": ("Latitude-CONE model (degrees)", "figures/correlation_lat.png"),
    "speed": ("Speed-CONE model (km/s)", "figures/correlation_speed.png"),
    "half_width": ("half width-CONE model (degrees)", "figures/correlation_half_width.png"),
}

# making and showing the data analysis plots:
sw.data_analysis_plots(cme_data, parameter_info)

############################################################################################################

# Machine Learning pipeline:
############################################################################################################
# defining the features for the machine learning model:
# normalize the cme_data:
cme_data = ml.normalize_data(cme_data, ["PE", "time_21_5"])
# save cme_data as a .xlsx file:
cme_data.reset_index(drop=True).to_excel("cme_data.xlsx", index=False)

# all the features:
features = ["B_diff", "P_diff"]

# uni results:
# calling the univariate model on these features:
uni_mae = ml.uni_LOOCV(cme_data, features)
print(uni_mae)
ml.uni_mae_plots(uni_mae, "figures/uni_mae_plots.png", cme_data)


# ranking the uni_features:
uni_ranked_features = ml.ranked_univariate(uni_mae, "figures/ranked_features.png")
print(uni_ranked_features)

"""
# multi results:
multi_mae = ml.multi_LOOCV(cme_data, features)
print(multi_mae)

# Getting the univariate and multivariate transit data:

print("These are the univariate ml results: ", uni_mae)
print("These are the multivariate results: ", multi_mae)

# plot this figure:
# ml.uni_mae_plots(uni_mae, multi_mae, "figures/ml_mae.png", cme_data, features)
"""