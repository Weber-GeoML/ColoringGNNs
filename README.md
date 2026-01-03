# Neural Algorithmic Reasoning for Approximate k-Coloring with Recursive Warm Starts

#### Knut Vanderbush and Melanie Weber

## Overview

This repository contains implementations of the algorithms `Mod-GCN`, `Full-GCN`, `Discrete-Color`, `Full-Color`, and `Triple-Color`, as well as the unmodified predecessor to `Mod-GCN`, described in the paper *Neural Algorithmic Reasoning for Approximate k-Coloring with Recursive Warm Starts.*

Each algorithm takes two arguments, a NetworkX graph `g` and a desired number of colors `k`, and can be run out of the box by supplying these arguments. Note that one must call 
```g = nx.convert_node_labels_to_integers(g)```
before running each algorithm, if the nodes of `g` are not already represented in this form. Letting `n` be the order of `g`, each algorithm outputs a list of `n` values between `0` and `(k - 1)`, where the `i`th value represents the color of vertex `i` in `g`. The function ``loss`` provided in each file takes two arguments, a NetworkX graph `g` and a list `colors` outputted by one of the coloring algorithms, and returns the loss of that coloring. The file `example.py` demonstrates how to use `Triple-Color` to color an Erdos-Renyi graph of order 200 and average degree 20 and display the loss of the coloring.

## Requirements

The code in this repository uses the Python libraries NetworkX, PyTorch, PyTorch Geometric, and NumPy.

## Unmodified GCN and Mod-GCN

The files `unmodified_gcn.py` and `mod_gcn.py` each contain the algorithm suggested by their name. The functions are named `unmod_gcn_color` and `mod_gcn_color` respectively.

## Discrete Algorithms

The file ``full_and_triple.py`` contains intuitive implementations of `Discrete-Color`, `Full-Color`, and `Triple-Color` as described in the paper. The functions are named `discrete_color`, `full_color`, and `triple_color` respectively. Here, `Discrete-Color` on a graph `g` of order `n` is computed by keeping track of a list of vertex colors and repeatedly computing a list of n values called `improvements`. The `i`th value in `improvements` is the maximum amount by which the loss of the current coloring can be decreased by changing the color of the `i`th vertex. Among all vertices with the maximum improvement value, one is selected uniformly at random, and then its color is changed to a new color uniformly at random among all colors that achieve the maximum decrease in loss. The `improvements` list is recomputed each time a vertex changes colors.

## Faster Full-Color and Triple-Color

The file ``full_and_triple_fast.py`` contains implementations of `Full-Color` and `Triple-Color` written in a way that is less intuitive, but runs more than three times faster for `Triple-Color`. As before, the functions are named `full_color` and `triple_color` respectively. Here, colorings of a graph of order `n` are stored in a `Coloring` object, which has attributes `adj_lst` being the adjacency list of the underlying graph, `k` being the number of colors used in the coloring, `colors` being the list of vertex colors, `neighbor_color_frequencies` being a `k`-by-`n` array whose `(i, j)`th entry is the number of neighbors of vertex `j` that have color `i`, `conflicts` being a list of `n` values whose `i`th value is the number of neighbors of vertex `i` that have the same color as vertex `i`, and `loss` being the loss of the coloring. Each `Coloring` object has a method called `improve`, which uses `Discrete-Color` to produce a `(k + 1)`-coloring based on the `Coloring` object's current `k`-coloring. The `improve` method updates the values in `colors`, `neighbor_color_frequencies`, `conflicts`, and `loss` each time a vertex changes colors. This cuts down on runtime because it avoids recomputing `improvements` from scratch each time a vertex changes color. It also cuts down on runtime for `Triple-Color` because it avoids computing `loss` from scratch for each final coloring. `Full-Color` is computed by calling `improve` on a `Coloring` object repeatedly until obtaining a `k`-coloring. `Triple-Color` is computed by calling `improve` repeatedly on `Coloring` objects, using a stack to simulate a depth-first search through the tree of colorings explored by `Triple-Color`.

## Full-GCN

The file `full_gcn.py` contains an implementation of `Full-GCN`. The function is named `full_gcn_color`. Here, graph colorings are again stored in a `Coloring` object. This time, the `improve` method follows the procedure for `Full-GCN` described in the paper.
