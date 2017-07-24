
# Installing TensorFlow in Linux OS using Anaconda

## Table of Contents
1. [Introduction](#introduction)

2. [Install Anaconda](#install_anaconda)

3. [Install tensorflow](#install_tensorflow)

4. [Validate Tensorflow Installation](#validate_tensorflow)

5. [Run on Jupyter Notebook](#run_notebook)

6. [References](#references)

<a id="introduction"></a>
## 1. Introduction

According to the its developers, Tensorflow is literally defined as:

>TensorFlow is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them. The flexible architecture allows you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API. TensorFlow was originally developed by researchers and engineers working on the Google Brain Team within Google's Machine Intelligence research organization for the purposes of conducting machine learning and deep neural networks research, but the system is general enough to be applicable in a wide variety of other domains as well.

There are different ways of installing tensorflow. These are:

   * virtualenv
   * "native" pip
   * Docker
   * Anaconda

In this tutorial, we are going to see the installation procedures using the fourth option, ```Anaconda```. We are going to install it in Ubuntu 64-bit OS. However, the installation procedures in other platforms are also similar.

You can get more information on the installation using different options from [here](https://www.tensorflow.org/install/)

So, if you are using a different operating system, you are expected to install and test TensorFlow before the class.

<a id=install_anaconda></a>
## 2. Install Anaconda

Before we install TensorFlow, we need to install ```anaconda```. It is an open source distribution of python simplifies package management and deployment. It can also be used for R programming language.

To get more information on anaconda, click [here](https://www.continuum.io/)

First go to anaconda [downloads page](https://www.continuum.io/downloads) and select the appropriate version

In our case, it is ubuntu 64-bit for python 2.7 version, the file will be similar to the following based on the current version of anaconda:

```
Anaconda2-4.3.1-Linux-x86_64.sh
```

To download the file, we have two ways:

* Click on the link in the web page(OR)
* Run the following command in a terminal

```
wget https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh 
```
    
Once it downloaded, go to the directory where the file was downloaded

Then type the following command to install it and follow the instructions:

```
bash Anaconda2-4.3.1-Linux-x86_64.sh
```

Once you install it, check if it is properly done by running the following command:
```
conda --version
```

You should get conda with its version number.

Note: if it is not yet recognized, it is most probably installed locally, so run the following command and check it again

```
source ~/.bashrc
```

<a id=install_tensorflow></a>
## 3. Install TensorFlow (CPU only):

One advantage of anaconda is that it enables you to create virtual environments so that you can install python packages separately without affecting packages in other environments.

You can get more information on managing conda environments from [here](https://conda.io/docs/using/envs.html)

Let's first see all the available conda environments by using the following command:

```
conda info --envs 
```

Note: the above command lists the available environments and puts ```*``` symbol to show the active one.

To make our life easier, we will create a new environment to install tensorflow on it.

We use the following command to create an environment named ```tensorflow```:

```
conda create -n tensorflow
```

Check if it is created by using the above command.

Once the environment is created, we have to activate it by using the following command:

```
source activate tensorflow
```

After the environment is activated, your command prompt should be change to show the active one.

Run the following commands to install TensorFlow inside your conda environment:

```
export TF_PYTHON_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.1-cp27-none-linux_x86_64.whl
```

```
pip install --ignore-installed --upgrade $TF_PYTHON_URL
```

where ```TF_PYTHON_URL``` is the URL of the TensorFlow Python package. You can check the latest binary file from [here](https://www.tensorflow.org/install/install_linux#the_url_of_the_tensorflow_python_package). In the above example, we selected the CPU-only version of TensorFlow for Python 2.7.


<a id=validate_tensorflow></a>
## 4. Validate TensorFlow Installation

Finally, we have to check if tensorflow is installed in our conda environment correctly. In this section, we will do validation check.

Before we run any program, we have to switch to the appropriate conda environment as follows:

```
source activate tensorflow
```

Note: to check the path and version of your current python installation, you can use the following commands:
```
which python
```

```
python -V
```

To run python from your terminal, type the following command:

```
python
```

Run the following simple python program in your terminal to check if TensorFlow is properly installed:

```
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
```

If the system outputs the following, then **!!!CONGRATS!!!** you are ready to begin running TensorFlow programs:

```Hello, TensorFlow!```

<a id=run_notebook></a>
## 5. Run on Jupyter Notebook

Python programs can be run on a terminal as shown in the above section to get quick results. However running on a command line is not comfortable for larger programs.

Another handy way of running is the Ipython Notebook. It is an web-based interactive environment where you type your code and display result on the same environment.

Fortunately, you don't need to install anything to use the feature. Because anaconda is prepackaged with the ```Jupyter Notebook``` tool.

To run the notebook, you follow these steps:

* Open a terminal

* Change to the directory where you want to put your ipython notebooks

* Activate your conda environment

```
source activate tensorflow
```

* Run either of the following commands to launch the jupyter notebook

```
jupyter-notebook .
```

```
jupyter notebook .
```
After you run the last command, a new jupyter notebook will be launched in your browser. The contents of you current directory will be displayed in the browser.

On your jupyter notebook, click on the following sequences of menus to start a new notebook.

```
File -> New Notebook -> Python 2
``` 

Then, type the following this program in to the new notebook cell:

```
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

After you typed your program in the cell, click on it and press ```Shift + Enter``` to run the program.

If you get the following output, then you have successfully run your first program on ipython notebook.

```
Hello, TensorFlow!
```

**Congrats!!!** You can now run your python programs in the notebook.

<a id=references></a>
## 6. References

[Tensorflow](https://www.tensorflow.org/)

[Installing Tensorflow](https://www.tensorflow.org/install/)

[Installing Tensorflow using Anaconda](https://www.tensorflow.org/install/install_linux#installing_with_anaconda)

[Anaconda Downloads](https://www.continuum.io/downloads)

[Managing Conda Environments](https://conda.io/docs/using/envs.html)
