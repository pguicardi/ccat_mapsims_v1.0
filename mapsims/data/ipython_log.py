# IPython log file

from astropy.table import QTable
f = QTable.read("planck_deltabandpass/planck_deltabandpass.tbl", format="ascii.ipac")
f.colnames
f
f.add_index("band")
f[70]["fwhm"]
f["70"]["fwhm"]
f.loc["70"]["fwhm"]
get_ipython().run_line_magic('logstart', '')
f[0]["fwhm"]
f.add_index("band")
f.loc["70"]["center_frequency"]
