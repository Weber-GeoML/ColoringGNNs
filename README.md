# Neural Algorithmic Reasoning for Approximate k-Coloring with Recursive Warm Starts

#### Knut Vanderbush and Melanie Weber

## Overview

This repository contains implementations of the algorithms `Mod-GCN`, `Full-GCN`, `Discrete-Color`, `Full-Color`, and `Triple-Color`, as well as the unmodified predecessor to `Mod-GCN`, described in the paper *Neural Algorithmic Reasoning for Approximate k-Coloring with Recursive Warm Starts* <a href="#1">[1]</a>. The paper is available at https://arxiv.org/abs/2601.05137.

Each algorithm takes two arguments, a NetworkX graph `g` and a desired number of colors `k`, and can be run out of the box by supplying these arguments. Note that one must call 
```
g = nx.convert_node_labels_to_integers(g)
```
before running each algorithm, if the nodes of `g` are not already represented in this form. Letting `n` be the order of `g`, each algorithm outputs a list of `n` integers between `0` and `k-1`, such that the `i`th integer is the color of vertex `i`. The function ``loss`` provided in each file takes two arguments, a NetworkX graph `g` and a list `colors` outputted by one of the coloring algorithms, and returns the loss of that coloring. The file `example.py` demonstrates how to use `Full-GCN` to color an Erdős–Rényi graph of order `n` and average degree `d`, then print the loss of the resulting coloring.

## Requirements

The code in this repository uses the Python libraries NetworkX, PyTorch, PyTorch Geometric, and NumPy.

## Unmodified GCN and Mod-GCN

The files `unmodified_gcn.py` and `mod_gcn.py` contain straightforward implementations of the unmodified GCN algorithm and `Mod-GCN` respectively as described in the paper. The functions are named `unmod_gcn_color` and `mod_gcn_color` respectively.

## Discrete Algorithms

The file ``full_and_triple.py`` contains straightforward implementations of `Discrete-Color`, `Full-Color`, and `Triple-Color` as described in the paper. The functions are named `discrete_color`, `full_color`, and `triple_color` respectively. Here, `Discrete-Color` on a graph of order `n` is computed as follows. We initialize a list `colors` of `n` integers between `0` and `k-1`, such that `colors[i]` is the color of vertex `i`; a list `neighbor_colors` of `n` dictionaries indexed by `0` through `k-1`, such that `neighbor_colors[i][c]` is the number of neighbors of vertex `i` that have color `c`; and a list `improvements` of length `n`, such that `improvements[i]` is the maximum amount by which the loss of the coloring may be reduced by changing the color of vertex `i`. Among all vertices with the maximum improvement value, one is selected uniformly at random, and its color is changed to a new color uniformly at random among all colors that achieve the maximum reduction in loss. Each time a vertex changes colors, the `neighbor_colors` dictionary for each of its neighbors is updated, and the `improvements` value for each of its neighbors is recomputed from scratch.

## Faster Full-Color and Triple-Color

The file ``full_and_triple_fast.py`` contains implementations of `Full-Color` and `Triple-Color` that are less intuitive, but run more than three times faster for `Triple-Color`. As before, the functions are named `full_color` and `triple_color` respectively. Here, colorings of a graph of order `n` are stored in a `Coloring` object, which has attributes `adj_lst` being a list of `n` lists, such that `adj_lst[i]` is the list of neighbors of vertex `i`; `k` being the number of colors used in the coloring; `colors` being a list of `n` integers between `0` and `k-1`, such that `colors[i]` is the color of vertex `i`; `neighbor_color_frequencies` being a `k`-by-`n` list of lists, such that `neighbor_color_frequencies[c][i]` is the number of neighbors of vertex `i` that have color `c`; `conflicts` being a list of length `n`, such that `conflicts[i]` is the number of neighbors of vertex `i` that have the same color as vertex `i`; and `loss` being the loss of the coloring.

Each `Coloring` object has a method called `improve`, which produces a `k+1` coloring via `Discrete-Color` using the current coloring as the starting colors. During `improve`, we keep track of a list `improvements` of length `n`, such that `improvements[i]` is the maximum amount by which the loss of the coloring may be reduced by changing the color of vertex `i`. The `improvements` list is initialized to be equal to `conflicts`, since initially, each vertex can eliminate all of its conflicts by switching to the new color. Among all vertices with the maximum improvement value, one is selected uniformly at random, and its color is changed to a new color uniformly at random among all colors that achieve the maximum reduction in loss. Each time a vertex changes colors, the values in `colors` and `loss` are updated, and for each neighbor of the selected vertex, the `neighbor_color_frequencies` values are updated, as is the `improvements` value and the `conflicts` value. This implementation cuts down on runtime by ensuring that `improvements` values are never computed from scratch, either at the start of a call of `Discrete-Color` or when a vertex changes colors. It also cuts down on runtime for `Triple-Color` by ensuring that `loss` is never computed from scratch, which is important in `Triple-Color` because we need to compute the loss of each final coloring in order to determine the best one.

`Full-Color` is computed by calling `improve` on a `Coloring` object repeatedly until obtaining a `k`-coloring. `Triple-Color` is computed by calling `improve` repeatedly on `Coloring` objects, using a stack to simulate a depth-first search through the tree of colorings explored by `Triple-Color`.

## Full-GCN

The file `full_gcn.py` contains a straightforward implementation of `Full-GCN` as described in the paper. The function is named `full_gcn_color`. Here, graph colorings are again stored in a `Coloring` object. This time, the `improve` method follows the procedure for `Full-GCN` described in the paper.

## References

<a id="1">[1]</a> Knut Vanderbush and Melanie Weber. *Neural Algorithmic Reasoning for Approximate k-Coloring with Recursive Warm Starts*. arXiv preprint arXiv:2601.05137, 2026. 
