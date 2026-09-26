from pathlib import Path
import argparse
import subprocess

from inspect_bids import inspect_subject


def run(command):
    print("\n$", " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True)


def register_anatomy(subject):
    info = inspect_subject(subject)
    derivatives = Path("derivatives") / subject



    derivatives = Path("derivatives") / subject
    anat_dir = derivatives / "anat"
    reg_dir = derivatives / "reg"
    mni_dir = reg_dir / "mni"
    func_dir = derivatives / "func"

    for directory in (anat_dir, reg_dir, mni_dir):
        directory.mkdir(parents=True, exist_ok=True)

    t1 = Path(info["t1w"])

    t1_brain = anat_dir / f"{subject}_T1w_brain.nii.gz"

    mc_mean = (
        func_dir
        / f"{subject}_task-heart_desc-stcMC_mean_bold.nii.gz"
    )

    epi2t1_prefix = reg_dir / f"{subject}_stcMC_epi2t1"
    epi2t1_mat = Path(str(epi2t1_prefix) + ".mat")
    epi2t1_image = Path(str(epi2t1_prefix) + ".nii.gz")

    mni_mat = mni_dir / f"{subject}_T1w_to_MNI_affine.mat"
    mni_image = mni_dir / f"{subject}_T1w_to_MNI_linear.nii.gz"

    mni_template = Path(
        subprocess.check_output(
            ["bash", "-lc", 'echo "$FSLDIR/data/standard/MNI152_T1_2mm.nii.gz"'],
            text=True,
        ).strip()
    )

    if not mc_mean.is_file():
        raise FileNotFoundError(f"Missing MCFLIRT mean: {mc_mean}")

    if not mni_template.is_file():
        raise FileNotFoundError(f"Missing MNI template: {mni_template}")

    # 1. Brain extraction
    if not t1_brain.exists():
        run([
            "bet",
            str(t1),
            str(t1_brain),
            "-R",
            "-f", "0.4",
            "-m",
        ])
    else:
        print("Existing T1 brain extraction:", t1_brain)


    # 2. EPI-to-T1 registration
    if epi2t1_mat.is_file() and epi2t1_image.is_file():
        print("Existing EPI-to-T1 registration:", epi2t1_mat)

    else:
        command = [
            "epi_reg",
            f"--epi={mc_mean}",
            f"--t1={t1}",
            f"--t1brain={t1_brain}",
        ]

        fieldmap = info["fieldmap"]

        if fieldmap is not None:
            fmap_dir = derivatives / "fmap"
            fmap_rads = fmap_dir / f"{subject}_fieldmap_rads.nii.gz"
            fmap_mag_brain = (
                fmap_dir / f"{subject}_magnitude1_brain.nii.gz"
            )

            if not fmap_rads.is_file() or not fmap_mag_brain.is_file():
                run([
                    "python",
                    "scripts/prepare_fieldmap.py",
                    subject,
                ])

            if not fmap_rads.is_file() or not fmap_mag_brain.is_file():
                raise RuntimeError("Fieldmap preparation failed")

            if info["phase_encoding"] != "j-":
                raise ValueError(
                    "Check phase-encoding direction before running epi_reg"
                )

            command.extend([
                f"--fmap={fmap_rads}",
                f"--fmapmag={fieldmap['magnitude1']}",
                f"--fmapmagbrain={fmap_mag_brain}",
                f"--echospacing={info['effective_echo_spacing']}",
                "--pedir=-y",
            ])

        command.append(f"--out={epi2t1_prefix}")
        run(command)


    # 3. Affine T1 to MNI registration
    if not mni_mat.exists() or not mni_image.exists():
        run([
            "flirt",
            "-in", str(t1),
            "-ref", str(mni_template),
            "-out", str(mni_image),
            "-omat", str(mni_mat),
            "-dof", "12",
            "-cost", "corratio",
        ])
    else:
        print("Existing T1-to-MNI registration:", mni_mat)

    print("\nSpatial registration complete:", subject)
    print("EPI-to-T1 matrix:", epi2t1_mat)
    print("T1-to-MNI matrix:", mni_mat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subject")
    args = parser.parse_args()

    register_anatomy(args.subject)
