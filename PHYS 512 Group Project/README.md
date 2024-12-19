# PHYS512_group_proj

Initial Plan:
Our plan is to simulate exoplanet transits/transit timing variations.

The general outline would be as follows:
1.⁠ simulate orbital dynamics of system w/ one planet around a Star; generate transit data and use mcmc to get best-fit parameters

2.⁠ ⁠Perturbation of lightcurve: introduce additional planet(s) and see what we can learn about the properties about the planet(s) in the system using mcmc (aka TTV)

MCMC on transit lightcurves:
- simulate lightcurve using starry --> then run MCMC (arbitrarily choose priors) to get params of planet from transit; we will see variation ; mcmc of each consecutive lightcurve (we expect mid-transit point to change --> aka TTVs)

second part:
- use TTVs --> run MCMC on those data to infer params of second body
