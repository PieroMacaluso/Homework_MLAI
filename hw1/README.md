**Homework 1 Report** **Piero Macaluso s252894**\
Machine Learning and Artificial Intelligence 2018/2019 Due Date:
19/01/2019\
Prof. Barbara Caputo

Data Preparation
================

In this first step, I was asked to prepare the images for the
elaboration. The dataset was made by $1087$ samples of $3$x$227$x$227$
size which belong to $4$ visual object categories. I extracted all the
files in the folder `PACS_homework` in the root folder of my workspace
then I used $4$ threads (one for every category) each one running . I
decided to save the images and the labels in two file (`data.npy` and
`label.npy`) in order to make next loadings faster.

Principal Component Visualization
=================================

I standardized `x` using `StandardScaler` making each feature zero-mean
and unit-variance. Then I applied PCA on the normalized X obtaining the
projection using all $1087$ PCs as in . I decided to implement a custom
function called `reconstruction()` in order to re-project `x_t` using
first $60$, $6$, $2$ and last $6$ principal component (PC) and visualize
the reconstructed images with `show_reconstruction()` function as you
can see in .

Comments about re-projections
-----------------------------

I tried to re-project a series of images from different categories and I
obtained the results in .

As we can see, we are able to distinguish some coarse details of the
original images in the re-projections with 60PC because these have a
cumulative variance of approximately $77\%$ but $6$PC and $2$PC plots
has lower variances because of the reduced number of PC used to
re-project.

The “Last $6$PC” one has very low variance (approximately $0\%$),
instead. I did expect this last result because of PCA transformation:
PCs are sort in descending order of variance, so last PCs can not be
able to describe correctly the whole dataset.

I noticed that all “Last $6$PC” plots describe the shape of a person:
this is probably because we have different amount of images for each
category and people ones are the majority. Indeed, the dataset is
composed by $189$ dogs ($\approx17\%$), $186$ guitars ($\approx17\%$),
$280$ houses ($\approx26\%$) and $432$ people ($\approx40\%$).

Furthermore, the great part of people images are from the same point of
view (frontal) in contrast to the other ones where we have different
points of view, numbers and rotation: maybe this could be another
underlying cause.

![Re-projections of a dog
image[]{data-label="fig:dog1"}](img/fig01a.png){height="0.5\paperwidth"}

![Re-projections of a person
image[]{data-label="fig:guitar1"}](img/fig01b.png){height="0.5\paperwidth"}

![Re-projections of a person
image[]{data-label="fig:house1"}](img/fig01c.png){height="0.5\paperwidth"}

![Re-projections of a person
image[]{data-label="fig:person1"}](img/fig01d.png){height="0.5\paperwidth"}

Comments about scatter and variance plots
-----------------------------------------

I produced scatter plot using different combination of PCs as we can see
in .

In there is a projection on $1$st and $2$nd PC: it is easy to find an
area for *person* and *house* classes, while *guitar* and *dog* ones
have a wider distribution on the plot.

In there are a $3$rd and $4$th PC projection and a $10$th and $11$th PC
one: in this case it seems more difficult to find crowded areas and
almost isolated of a single category.

I noticed that the range of coordinates in the graphs was about
inversely proportional to the position of PC used, or, better, that the
points in the scatter plot were getting closer and closer to the mean
(which is $0$ because of normalization) resulting in **decreased
variance**. This can be explained from theoretical perspective because,
in the definition of PCA transformation, the first principal component
has the **largest possible variance**, and each succeeding component in
turn has the highest variance possible under the constraint that it is
**orthogonal** to the preceding components.

Plotting the cumulative sum of variance ratio (provided in the PCA of
`sklearn`) may be useful in order to decide how many components are
necessary to preserve data without much distortion. In general, there is
not a right and unique way to select the *correct* number of Principal
Components, but in this context we can analyze the graph and select a
number of PC characterized by a cumulative sum of variance ratio higher
than a specific threshold (e.g., $95\%$).

![Projection on $1$st and $2$nd
PC[]{data-label="fig:scatter1"}](img/fig02a.png){height="0.5\paperwidth"}

![Projection on $3$rd and $4$th
PC[]{data-label="fig:scatter2"}](img/fig02b.png){height="0.5\paperwidth"}

![Projection on $10$th and $11$th
PC[]{data-label="fig:scatter3"}](img/fig02c.png){height="0.5\paperwidth"}

![Projection on $1$st, $2$nd and $3$rd
PC[]{data-label="fig:scatter4"}](img/fig02d.png){height="0.5\paperwidth"}

![Variance and Cumulative
Variance[]{data-label="fig:variance"}](img/fig03.png){height="0.4\paperwidth"}

Classification
==============

The formulation of Naïve Bayes is described in \[eq:1\]. $$\label{eq:1}
        \hat{y}=\operatorname*{argmax}_{i \in \{1,\dots,k\}} p ( y_i \mid x_1, \dots, x_d) = \operatorname*{argmax}_{i \in \{1,\dots,k\}}\overbrace{p(y_i)}^{Prior} \prod_{j=1}^{d}\overbrace{p(x_j\mid y_i)}^{Likelihood}$$
where

& predicted label\
y\_i & $i$-th label with $i \in \{1,\dots,k\}$\
x\_j & $j$-th example with $j \in \{1,\dots,d\}$\
p(xy) & Gaussian

Firstly, I splitted randomly `x_n` (`x` standardized) and the labels in
**training** ($75$%) and **test set** ($25$%), then I applied
`GaussianNB` of `sklearn` obtaining an accuracy of $\boldsymbol{75\%}$.

After that, I splitted in the same way `x_t` (the projection of `x_n`
obtained from PCA transformation) and I used these sets to train and
test the classifier first using the data projected on $1$st and $2$nd PC
and finally the one projected on $3$rd and $4$th PC. In the first case
the accuracy was $\boldsymbol{59.19\%}$, while in the second one was
$\boldsymbol{48.53\%}$.

I plotted **decision boundaries** of the classifier and the projection
of the whole dataset for both cases as we can see in .

Comparing accuracy results, we can conclude that classifications done
after the application of PCA are worse than the one on the original
normalized dataset. Furthermore, the accuracy of classifier on $1$st and
$2$nd PC is higher than the one of classifier on $3$rd and $4$th PC,
probably because of the different variance that a specific component can
describe. This may be due to the fact that reducing dimensionality may
discard important information useful to discriminate a class from the
others in a classifier.

![Decision boundaries of first classifier on $1$st and $2$nd
PC[]{data-label="fig:class1"}](img/fig04.png){height="0.52\paperwidth"}

![Decision boundaries of classifier on $3$rd and $4$th
PC[]{data-label="fig:class2"}](img/fig05.png){height="0.52\paperwidth"}

Code Execution
==============

Requirements
------------

-   Python 3

-   All dependencies in `requirements.txt`.

    `$ pip install -r requirements.txt` to install them

Usage
-----

-   `$ python main.py -n <PACS_homework folder>`

    Loads Image from the specified `<PACS_homework folder>` and execute
    the code

-   `$ python main.py -s <PACS_homework folder>`

    Loads Image from the specified `<PACS_homework folder>` and save
    files in `data.npy` and `label.npy` for faster next execution before
    executing the code

-   `$ python main.py -l <data.npy file> <label.npy file> `

    Loads Image from `data.npy` and `label.npy` files and execute the
    code

Reproducibility
---------------

In order to reproduce the same data for this experiment you have to
change the global variable `r_state` (line) from `None` to $252894$
which is my badge number.

Attachments {#attachments .unnumbered}
===========

-   Source Code:

    -   `main.py`

    -   `requirements.txt`
