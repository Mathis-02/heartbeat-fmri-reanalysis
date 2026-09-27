# Analysis workflow

These notes describe the main analysis pipeline I used for the heartbeat fMRI reanalysis. The pipeline changed a few times while I was testing the preprocessing and registration steps, so this document mainly describes the final choices.

## Data

The data come from OpenNeuro dataset `ds003763`.

The functional task compares two conditions:

- `Heart`: attention directed towards heartbeats
- `Sound`: attention directed towards external sounds

Each run contains 16 blocks, with 8 blocks per condition. Blocks last 20 seconds and the two conditions alternate throughout the run.

The BOLD data have 259 volumes with a TR of 2 s. I worked mainly with 10 subjects during preprocessing and first-level QC.

## Initial inspection

Before preprocessing, I used NiBabel and the BIDS metadata to check the dimensions, voxel sizes, orientation and acquisition parameters of the images.

For the functional images, I also checked the events files and slice-timing information before applying any correction.

This part was useful for keeping track of the different coordinate systems involved in the analysis. In particular, I kept the native EPI space separate from anatomical T1 space and MNI template space rather than treating registration as a change in the image labels only.

## Functional preprocessing

Slice-timing correction and motion correction were applied to the BOLD series. The resulting files are identified with `stcMC` in the filenames.

Motion parameters were inspected for each subject. I calculated framewise displacement and used it as part of the subject-level QC. Some subjects showed substantially more movement than others, so I also implemented a version of the first-level model with additional spike regressors as a sensitivity check.

Fieldmaps were available for most subjects. They were prepared separately and used during EPI-to-T1 registration when available.

The preprocessing scripts are kept separately rather than wrapped into a single external pipeline because I wanted to inspect the intermediate files and understand what each transformation was doing.

## First-level GLM

The first-level GLM was fitted in native functional space.

The design matrix contains regressors for the Heart and Sound conditions together with motion and drift terms. The main contrast used in the later analyses is:

`Heart - Sound`

For each subject I saved the contrast effect estimate as well as the corresponding t and z maps.

I checked the design matrices and their condition numbers and compared the main model with additional versions used during QC, including a model with spike regressors for higher-motion volumes.

The effect estimate, rather than the thresholded z map, was used for the later group analysis.

## Registration and spatial normalization

EPI-to-T1 and T1-to-MNI registration were treated as two separate steps.

For EPI-to-T1 registration I used FSL. When a fieldmap was available, it was incorporated into `epi_reg` to account for susceptibility distortions. I visually checked the resulting alignment and also tested alternative registration procedures for problematic cases.

For T1-to-MNI registration I first used an affine FSL transformation. I also tested FNIRT for nonlinear registration, but one of the tests produced negative Jacobian values, indicating folding in the deformation field. I did not retain this solution.

I then implemented a separate ANTs registration using rigid, affine and SyN transformations. The deformation fields were checked for folding before being used in the later analysis.

For the final ANTs branch, subject-level contrast maps were transformed in two steps:

1. native EPI → subject T1 using the FSL EPI-to-T1 transformation
2. subject T1 → MNI152 2 mm using the ANTs affine and nonlinear transformations

I used the same approach for native functional masks, with nearest-neighbour interpolation for the masks.

One subject (`sub-09381`) had required a specific EPI-to-T1 correction during the earlier FSL analysis. Rather than mixing this procedure with the registration used for the other subjects, I excluded this subject from the later ANTs/group-analysis branch. The earlier preprocessing work included 10 subjects, whereas the final ANTs/group-analysis branch used 9 subjects.

This left 9 subjects for the final ANTs branch.