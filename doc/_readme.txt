ReadTheDocs
The main documentation of rtc-tools that is accessible for all users can be found on rtc-tools.readthedocs.io.

By default, the documentation corresponds to the RTC-Tools version of the latest release. At the bottom-left, you can choose between three versions of rtc-tools: latest (the remote master branch), or stable (the latest release). 

The source code for rtc-tools.readthedocs.io is in the doc folder of the RTC-Tools repository: rtc-tools . To build the documentation locally, you sphinx and sphinx-rtd-theme must be installed. 

Build the documentation locally

Prepare a Python virtual environment and install packages
If you don't have a virtual environment ready, navigate to the folder "rtc-tools" and create a virtual environment there:
	python -m venv venv

Activate your virtual environment:
	venv\scripts\activate.bat

Navigate to the folder "doc" 
	cd doc

If not yet installed, install the necessary Python packages by executing the following command:
	pip install -r requirements.txt

Now install the local copy of the RTC-Tools repository you are working in as editable module in the virtual environment. Execute the code
	pip install -e <local path>rtc-tools\src\rtctools\
Where 
	<local path>
points to the current directory, where the local clone of the RTC-Tools repository is located. 

Build the documentation

To build the documentation, in the virtual environment, execute 
	make html
in your virtual environment.

Optionally to force a rebuild of the documentation:
	make clean | make html
This will generate html files in the _build/html folder. To see the documentation, open the _build/html/index.html file in a browser.