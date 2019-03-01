##  Setting Up

Begin by downloading Anaconda (Python 3.7 Version):

    https://www.anaconda.com/download/


Run these commands (Anaconda Prompt for Windows, Terminal for Mac):

    conda create --name system_diagnostics (only if you have not already done this step once before)
    
    conda install nodejs (only if you have not already done this step once before)

    source activate system_diagnostics (activate system_diagnostics if Windows/Linux)


Download this repo, then navigate to the diagnostics directory, located within the System_Diagnostics folder, then run:

    conda install --file requirements.txt (only if you have not already done this step once before)

    bokeh serve --show .
