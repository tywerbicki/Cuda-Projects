# CUDA Projects

A place for my personal CUDA tinkering to reside. The layout of the repository adheres to [The Pitchfork Layout](https://api.csswg.org/bikeshed/?force=1&url=https://raw.githubusercontent.com/vector-of-bool/pitchfork/develop/data/spec.bs), specifically using [seperate header placement](https://api.csswg.org/bikeshed/?force=1&url=https://raw.githubusercontent.com/vector-of-bool/pitchfork/develop/data/spec.bs#src.header-placement).

### Dependencies ###
* [CMake](https://cmake.org/download/)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

### Perform Out-Of-Source Build ###

From the repository root:
```
mkdir build && cd build && cmake ..
```
For example, to build *Tests* for *Release* on Windows:
```
msbuild.exe Tests.vcxproj /property:Platform=x64 /property:Configuration=Release
```
To then run the *Tests* executable:
```
Release\Tests.exe
```

---

## Saxpy ##

The canonical *single-precision ax + y kernel*:
$$\mathbf{\overline{y}} = \alpha\mathbf{\overline{x}} + \mathbf{\overline{y}} \quad \textrm{where} \quad \mathbf{\overline{x}}, \mathbf{\overline{y}} \in \mathbb{R}^{n}, \textrm{ } \alpha \in \mathbb{R}$$
This is used as a prototype kernel for code and file organization.

---

## Utilities ##

CUDA and host utility functions that are useful when working with the CUDA programming model.

---
