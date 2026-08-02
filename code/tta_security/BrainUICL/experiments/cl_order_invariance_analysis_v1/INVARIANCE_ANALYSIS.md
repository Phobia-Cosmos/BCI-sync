# EEG Invariance Structure Audit

Lower normalized total standard deviation means the normalized statistic is more stable across sampled sequences and subjects. Cross-dataset similarity uses a standardized mean difference below 0.5 only as a screening rule, not proof of physiological equivalence.

| Component | ISRUC normalized std | FACED normalized std | Cross-dataset effect size | Similar screen |
|---|---:|---:|---:|---:|
| delta_relative_power | 0.1815 | 0.0837 | 0.1014 | yes |
| theta_relative_power | 0.5469 | 0.1895 | 0.0669 | yes |
| alpha_relative_power | 0.8066 | 0.5079 | 0.1876 | yes |
| beta_relative_power | 1.2922 | 0.2271 | 0.0898 | yes |
| gamma_relative_power | 0.2220 | 0.1477 | 0.5752 | no |
| covariance_eigen_1 | 0.2071 | 0.1086 | 3.0658 | no |
| covariance_eigen_2 | 0.3186 | 0.1073 | 0.8597 | no |
| covariance_eigen_3 | 0.3834 | 0.1549 | 2.8441 | no |
| covariance_eigen_4 | 0.4534 | 0.2459 | 1.6613 | no |
| covariance_eigen_5 | 0.2067 | 0.1542 | 3.7245 | no |
| covariance_eigen_6 | 0.1166 | 0.1412 | 4.5224 | no |
| autocorrelation_10ms | 0.2097 | 0.0921 | 0.1014 | yes |
| autocorrelation_40ms | 0.3416 | 0.1914 | 0.2400 | yes |
| autocorrelation_160ms | 0.5459 | 0.2423 | 0.5179 | no |
