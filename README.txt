ADDITIONS/CHANGES TO THE MAPSIMS MODULE:
    - noise.py
        - _get_requested_hitmaps
        - get_survey
        - get_hitmaps
        - import section
    - data/simonsobs_instrument_parameters_2020.06
        - input data for 5 CCAT channels were added to simonsobs_instrument_parameters_2020.06.tbl
        - 5 bandpass files were added, each corresponding correspond to a new CCAT channel

DEPENDENCIES:
    - ALL MAPSIMS DEPENDENCIES
    - astropy
    - emcee
    - scipy
    - corner
    - pysm3