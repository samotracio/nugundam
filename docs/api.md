# API reference

The API below is generated directly from the **0.7.1** NumPy-style source docstrings with `mkdocstrings`. Normal workflows import public objects from the top-level `nugundam` namespace; source module paths are shown to make the implementation location explicit.

## Angular measurements

::: nugundam.angular.api.acf

::: nugundam.angular.api.accf

::: nugundam.angular.api.ang_auto_counts

::: nugundam.angular.api.ang_cross_counts

## Projected measurements

::: nugundam.projected.api.pcf

::: nugundam.projected.api.pccf

::: nugundam.projected.api.proj_auto_counts

::: nugundam.projected.api.proj_cross_counts

## Marked measurements

::: nugundam.marked.macf

::: nugundam.marked.maccf

::: nugundam.marked.mpcf

::: nugundam.marked.mpccf

## Angular configuration

::: nugundam.angular.models.CatalogColumns

::: nugundam.angular.models.AngularBinning

::: nugundam.angular.models.AngularGridSpec

::: nugundam.angular.models.WeightSpec

::: nugundam.angular.models.BootstrapSpec

::: nugundam.angular.models.JackknifeSpec

::: nugundam.angular.models.ProgressSpec

::: nugundam.angular.models.SplitRandomSpec

::: nugundam.angular.models.AngularAutoConfig

::: nugundam.angular.models.AngularCrossConfig

::: nugundam.angular.models.AngularAutoCountsConfig

::: nugundam.angular.models.AngularCrossCountsConfig

## Projected configuration

::: nugundam.projected.models.ProjectedCatalogColumns

::: nugundam.projected.models.ProjectedBinning

::: nugundam.projected.models.ProjectedGridSpec

::: nugundam.projected.models.DistanceSpec

::: nugundam.projected.models.ProjectedPdfSpec

::: nugundam.projected.models.PDFSourceSpec

::: nugundam.projected.models.ProjectedPdfSourceSpec

::: nugundam.projected.models.ProjectedMCPdfSpec

::: nugundam.projected.models.ProjectedAutoConfig

::: nugundam.projected.models.ProjectedCrossConfig

::: nugundam.projected.models.ProjectedAutoCountsConfig

::: nugundam.projected.models.ProjectedCrossCountsConfig

## Mark specifications

::: nugundam.marked.AutoMarkSpec

::: nugundam.marked.CrossMarkSpec

## Result and count classes

::: nugundam.angular.models.AngularAutoCounts

::: nugundam.angular.models.AngularCrossCounts

::: nugundam.angular.models.AngularAutoCountsResult

::: nugundam.angular.models.AngularCrossCountsResult

::: nugundam.angular.models.AngularCorrelationResult

::: nugundam.projected.models.PreparedProjectedSample

::: nugundam.projected.models.ProjectedAutoCounts

::: nugundam.projected.models.ProjectedCrossCounts

::: nugundam.projected.models.ProjectedAutoCountsResult

::: nugundam.projected.models.ProjectedCrossCountsResult

::: nugundam.projected.models.ProjectedCorrelationResult

::: nugundam.marked.MarkedAngularCorrelationResult

::: nugundam.marked.MarkedProjectedCorrelationResult

## PDF tools

::: nugundam.projected.pdf_tools.PiMaxEstimate

::: nugundam.projected.pdf_tools.CompressedPdfGMM

::: nugundam.projected.pdf_tools.estimate_pi_max_from_pdfs

::: nugundam.projected.pdf_tools.compress_pdfs_to_gmm

::: nugundam.projected.pdf_tools.plot_gmm_for_object

## Pair diagnostics

::: nugundam.projected.diagnostics.summarize_pair_diagnostics

::: nugundam.projected.diagnostics.print_pair_diagnostics

## Estimator reconstruction

::: nugundam.angular.estimators.compute_auto_wtheta

::: nugundam.angular.estimators.compute_cross_wtheta

::: nugundam.projected.estimators.compute_auto_xi2d

::: nugundam.projected.estimators.compute_cross_xi2d

## I/O

::: nugundam.io.write_result

::: nugundam.io.read_result

::: nugundam.io.save_result

::: nugundam.ascii_io.write_ascii

::: nugundam.angular_public.result_to_dict

## Plotting

::: nugundam.plotting.plotcf

::: nugundam.plotting.plot_result

::: nugundam.plotting.plotcf2d

::: nugundam.plotting.plot_result2d

::: nugundam.plotting.plot_compare_ratio

::: nugundam.plotting.plot_cov_matrix

::: nugundam.plotting.plot_corr_matrix

::: nugundam.plotting.plot_jk_regions
