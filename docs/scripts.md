# Scripts

Most scripts are run from the root of the repository. Intermediate images and QC files are written to `derivatives/`, which is not tracked by Git.

## Main workflow

`inspect_bids.py`
Checks the files and metadata required for each subject.

`apply_slice_timing.py`
Applies slice-timing correction to the functional run.

`prepare_fieldmap.py`
Prepares the fieldmap files used during EPI-to-T1 registration.

`register_anatomy.py`
Runs the FSL anatomical and EPI-to-T1 registration steps.

`run_pipeline.py`
Runs the main subject-level preprocessing and analysis workflow.

`run_batch.py`
Runs the subject-level pipeline over several subjects.

`build_design_matrix.py`
Builds and saves the first-level design matrix.

`fit_first_level_glm.py`
Fits the first-level GLM and computes the Heart - Sound contrast.

## Registration and group analysis

`register_ants.py`
Runs rigid, affine and SyN T1-to-MNI registration with ANTs and checks the resulting deformation.

`build_native_masks.py`
Creates functional masks from the preprocessed BOLD data in native EPI space.

`transform_masks_to_mni.py`
Transforms the native functional masks through T1 space to MNI space.

`build_common_mask_mni.py`
Builds the strict common mask used for the final 9-subject group analysis.

`transform_contrasts_to_mni.py`
Transforms the first-level Heart - Sound effect maps to MNI space.

`group_analysis.py`
Runs the voxelwise one-sample group analysis and FDR correction.

## MVPA

`mvpa_extract_blocks.py`
Builds block-level multivoxel patterns, runs the cross-validation and computes the within-pair permutation distributions.

`mvpa_group_analysis.py`
Uses the subject-level permutation results to construct the group null distribution.

`plot_mvpa_results.py`
Produces the MVPA summary figures.

## QC and sensitivity analyses

The remaining scripts were used during development to inspect intermediate results or test specific processing choices. These include motion and alignment checks, fieldmap/no-fieldmap comparisons, spike-regressor models, coverage comparisons and alternative EPI registration tests.

They are kept in the repository because they document the QC steps and some of the decisions made while developing the final workflow.

## FreeSurfer

`freesurfer/extract_surface_metrics.py` reads precomputed `fsaverage` surface data and summarizes cortical thickness by `aparc` region.

This is separate from the main fMRI pipeline and does not involve subject-specific `recon-all` processing.