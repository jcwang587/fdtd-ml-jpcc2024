# fdtd-ml-jpcc2024

This repository contains the machine learning code used in the publication *Machine Learning-Assisted Light Management and Electromagnetic Field Modulation of Large-Area Plasmonic Coaxial Cylindrical Pillar/Ring Nanoarray Patterns*.

## Introduction

As required by the program, the data calculated from the FDTD simulation is not accessible. You can substitute the input CSV with your own file as needed for your project, including input parameters and either a single or multiple targets for machine learning model training. 

`xgboost_opt_e`: Training script for the target electric field enhancement.

`xgboost_opt_q`: Training script for the target Q-factor.

 `xgboost_opt_multi`: Training script for (1) the multiple targets electric field enhancement and Q-factor, and (2) SHAP analysis for model explanation.

If you have any questions regarding the code or setting up your project, please feel free to leave a message in the issues channel.

## Reference

The original paper provides detailed information on the project's background and experimental content.

```bibtex
@article{JPCC2024,
  title = {Machine Learning-Assisted Light Management and Electromagnetic Field Modulation of Large-Area Plasmonic Coaxial Cylindrical Pillar/Ring Nanoarray Patterns},
  author = {Wang, Anyang and Hang, Yingjie and Wang, Jiacheng and Tan, Weirui and Wu, Nianqiang},
  journal = {The Journal of Physical Chemistry C},
  volume = {128},
  number = {30},
  pages = {12495--12502},
  numpages = {6},
  year = {2024},
  month = {6},
  publisher = {American Chemical Society (ACS)},
  doi = {10.1021/acs.jpcc.4c01405},
  url = {http://dx.doi.org/10.1021/acs.jpcc.4c01405}
}
```

