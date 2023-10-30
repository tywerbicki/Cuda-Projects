## CUDA Projects

A place for my personal CUDA tinkering to reside. The layout of the repository adheres to [The Pitchfork Layout](https://api.csswg.org/bikeshed/?force=1&url=https://raw.githubusercontent.com/vector-of-bool/pitchfork/develop/data/spec.bs), specifically using [seperate header placement](https://api.csswg.org/bikeshed/?force=1&url=https://raw.githubusercontent.com/vector-of-bool/pitchfork/develop/data/spec.bs#src.header-placement).

---

### Saxpy ###

The canonical *single-precision ax + y kernel*:
$$\mathbf{\overline{y}} = \alpha\mathbf{\overline{x}} + \mathbf{\overline{y}} \quad \textrm{where} \quad \mathbf{\overline{x}}, \mathbf{\overline{y}} \in \mathbb{R}^{n}, \textrm{ } \alpha \in \mathbb{R}$$
This is used as a prototype kernel for code and file organization.

---

### Utilities ###

CUDA and host utility functions that are useful when working with the CUDA programming model.

---
