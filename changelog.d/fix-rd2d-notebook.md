#### RD_2D example notebook is tree-level

The d=2 reaction–diffusion example asked for one loop and for the equal-point value at the origin; both are refused since the UV-divergence checks (the tadpole is log-divergent at d=2 and needs a mass counterterm). The notebook now computes the tree on a grid that starts at r=0.25, and shows the refusal (`SpatialUVDivergenceError`) as a documented scope limit. This was the failure of the weekly notebooks workflow on Oct 19.
