This folder contains the Python scripts used for the spectrum-processing branch of the project.

Purpose:

The scripts in this folder read BearinX spectrum files, inspect their structure, extract spectrum values, calculate RMS values for defined frequency bands, and prepare a machine-learning dataset.

This branch was exploratory and was separate from the final G28/G200 benchmark.

Main scripts:

1. inspect_spectrum_file.py

Opens and inspects a spectrum file.

It was used to understand:

* the file structure
* the header layout
* where the numerical spectrum data begin
* how the frequency and amplitude values are stored

2. find_spectrum_header_marker.py

Searches the spectrum file for the text or marker that identifies the end of the header and the beginning of the numerical data.

3. parse_spectrum_header.py

Reads useful information from the spectrum-file header.

This may include:

* signal names
* units
* metadata
* spectrum settings

4. inspect_frequency_axis.py

Checks the frequency-axis values.

It was used to confirm:

* the frequency range
* the frequency step
* whether the frequency values were read correctly

5. extract_spectrum_table.py

Extracts the numerical spectrum table from the original spectrum file.

The extracted table contains frequency values and the corresponding spectrum amplitudes.

6. compute_band_rms_single.py

Calculates RMS values for one spectrum file.

The spectrum is divided into defined frequency ranges such as:

* low-frequency band
* medium-frequency band
* high-frequency band
* broadband range

7. compute_band_rms_all.py

Repeats the band-RMS calculation for all available spectrum files.

The resulting values are combined into one output table.

8. build_dataset_spectrum.py

Combines:

* the bearing input parameters
* the calculated spectrum-band RMS values

This creates the machine-learning dataset used for the spectrum-learning experiments.

Typical workflow order:

1. inspect the spectrum-file structure
2. identify and parse the header
3. inspect the frequency axis
4. extract the spectrum table
5. calculate RMS values for one file
6. calculate RMS values for all files
7. build the final spectrum-learning dataset

Project role:

This folder belongs to the earlier framework-development phase.

The generated spectrum results are stored under:

data/outputs/spectrum

The final G28/G200 custom-versus-AutoGluon benchmark is stored separately under:

scripts/new_data_comparison