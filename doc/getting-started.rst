Getting Started
+++++++++++++++

Installation
============

Python 3.11 or newer (up to 3.15) is recommended for running RTC-Tools.

For most users, the easiest way to install RTC-Tools using the `pip <https://pip.pypa.io/>`_ package manager.

Using the Pip Package Manager
-----------------------------

Although not required, it is recommended to install RTC-Tools in a virtual
environment. See the `official Python tutorial
<https://docs.python.org/3/tutorial/venv.html>`_ for more information on how
to set up and activate a virtual environment.

RTC-Tools, including its dependencies, can be installed using the `pip <https://pip.pypa.io/>`_ package manager::

    # Install RTC-Tools using pip package manager
    pip install rtc-tools

From Source
-----------

The latest RTC-Tools source can be downloaded using git::

    # Get RTC-Tools source
    git clone https://github.com/rtc-tools/rtc-tools.git

Then you can install this latest version as follows::

    pip install ./rtc-tools

Or if you would like to have an editable installation (e.g. as developer)::

    pip install -e ./rtc-tools

Note that rtc-tools-channel-flow is a dependency of rtc-tools which is included in the above installations. 

.. _getting-started-download-examples:

Downloading and running examples
================================

To check whether the installation was succesful, the basic example can be
used. If RTC-Tools was not installed from source, the examples need to be
downloaded first::

    # Download the examples to the current folder (.)
    rtc-tools-download-examples .

    # Navigate to the basic example
    cd rtc-tools-examples/basic/src

    # Run the example
    python example.py

If the installation was succesful, you should see that the solver succeeds:

.. image:: images/basic_example_console.png

Elsewhere in this documentation we refer to the folder containing the examples
as ``<examples directory>``. Depending on the method of installation this can
then either be:

    * ``\path\to\rtc-tools-examples``, when having downloaded the examples
    * ``\path\to\source\of\rtc-tools\examples``, when having installed RTC-Tools from source

.. _getting-started-copy-libraries:

Copying Modelica libraries
==========================

Because the Modelica libraries are distributed as pip packages, their location
inside Python's site-packages can be somewhat inconvient. To copy the Modelica
libraries to a more convenient location, you can use the ``rtc-tools-copy-libraries``
command::

    # Copy all Modelica libraries of RTC-Tools to the current folder (.)
    rtc-tools-copy-libraries .

You should now have a folder ``Deltares``, containing amongst others a
``package.mo`` file, a ``ChannelFlow`` folder and folders of any other RTC-
Tools extensions you installed.

Elsewhere in this documentation we refer to the library folder containing the
``Deltares`` folder as ``<library directory>``.


.. _getting-started-omedit:

Getting OMEdit
==============

RTC-Tools uses the Modelica language to describe the mathematics of the system
we wish to optimize. There are several editors for Modelica models, but the
OpenModelica Connection Editor, or OMEdit, is a free and open-source graphical
connection editor that can be used to construct RTC-Tools models. The latest
compatible version is the 1.21 release. To download it for Windows, click here:
https://build.openmodelica.org/omc/builds/windows/releases/1.21/



Once installed, you can start OMEdit by clicking::

   Start -> All Programs -> OpenModelica -> OpenModelica Connection Editor

With OMEdit installed, you can start using it by following along with the basic
example, :doc:`examples/optimization/basic`.

.. note::
    
        The example models are based on version 4.0 of the modelica standard library.
        Make sure you load this library when editing the models in OMEdit.

.. _running-rtc-tools:

Running RTC-Tools
=================

RTC-Tools is run from a command line shell. On Windows, both ``PowerShell``
and ``cmd`` can be used. On Linux/MacOS you could use the terminal application
with a shell of your liking.

Once you have started the shell and loaded the correct virtual environment (if
applicable), navigate to the ``src`` directory of the case you wish to
optimize, e.g.::

    cd \path\to\rtc-tools-examples\basic\src

Then, to run the case with RTC-Tools, run the ``src`` python script, e.g.::

    python example.py

You will see the progress of RTC-Tools in your shell. All your standard shell
commands can be used in the RTC-Tools shell. For example, you can use::

    python example.py > log.txt

to pipe RTC-Tools output to a log file.
