Setting up a Virtual Environment for Jupyter on MACOS

To set up a virtual environment on MACOS, open Terminal. Now type 

- cd (directory)
  - (directory) is the location of the folder you want this virtual environment to exist in
  - For me, this is 
    - cd /Users/michaelgiordano/Documents/GitHub/OCR\_Improvements/OCR\_Python
- mkdir venv
  - This creates a new folder called “venv” in the directory given before
- cd venv
  - This moves you into the new folder just created
- ls
  - This will list any files that might be inside the new folder. This should be blank
- pip install virtualenv
  - This will install the Python package that will allow you to generate the new virtual environment. This can be skipped if already installed.
- virtualenv venv -p python3
  - This installs python3 in the new folder and creates the virtual environment
- source venv/bin/activate
  - This activates the virtual environment. Now you are using the new virtual environment
  - To enter into this virtual environment from now on, use
    - source (directory)/venv/bin/activate
      - For me, this is
  - source /Users/michaelgiordano/Documents/GitHub/OCR\_Improvements/OCR\_Python/venv/bin/activate

Now you can install all of the necessary packages. Using the last step from above, activate your venv. Now you can import all of the packages we used, along with the versions used at the time of creation by using the requirements.txt file. In your Terminal, type

- pip install -r (directory)/requirements.txt
  - For me, this was
    - pip install -r /Users/michaelgiordano/Documents/GitHub/OCR\_Improvements/OCR\_Python/requirements.txt

Now you need to set up this virtual environment for use with Jupyter Notebooks. In your active virtual environment, type

- ipython kernel install --user --name=venv
  - or instead of venv, whatever you called your new folder above

Now you can run a Jupyter Notebook in this new virtual environment. In your active virtual environment, type

- jupyter notebook

Now you can open up a jupyter notebook within this virtual environment.
