# desc-hack-2026

Notebooks for finding transients and their host galaxies in the [OpenUniverse2024](https://doi.org/10.1093/mnras/staf1833) simulated Roman and Rubin surveys, cutting out their images, and measuring the transients' light curves by modelling each scene jointly across epochs and instruments with [scarlet2](https://github.com/pmelchior/scarlet2).

Those notebooks were written for this project. The repository also includes the OpenUniverse2024 tutorials written by IRSA ([Caltech-IPAC/irsa-tutorials](https://github.com/Caltech-IPAC/irsa-tutorials)), so they can be rerun on their own; [`NOTEBOOKS.md`](NOTEBOOKS.md) lists which notebook is which.
All data are read from the public `s3://nasa-irsa-simulations/` bucket and IRSA's simulated-data image search, so no AWS account or local data download is needed.

- **What each notebook and script does:** [`NOTEBOOKS.md`](NOTEBOOKS.md)
- **Setting up the environment:** [`setup_on_mac.md`](setup_on_mac.md)

## License

BSD 3-Clause, from the original repository. See `LICENSE`.
