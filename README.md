# t-distributed stochastic neighbor embedding (t-SNE)

Implementation of the popular t-SNE algorithm (van der Maaten & Hinton [http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf](2008)).

Please download _train-images-idx3-ubyte_ and _train-labels-idx1-ubyte_ from http://yann.lecun.com/exdb/mnist/ and place the files in the same folder as the tsne executable.

Execute with **./tsne SAMPLES THREADS**. For example: ./tsne 15000 4 will run t-SNE on a sample of size 15000, utilizing 4 threads. Use all images with ./tsne 60000 or any invalid value (like ./tsne 0).

Executing once will create a default config file. After this, debugging is easy by changing the values in the file. Make sure that the format stays as "VAR\<space\>=\<space\>VALUE". (For now, refer to /include/tsne/util.hpp for details.)