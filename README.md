# CUDA Lab

There is a live website of this project at [http://www.aleacubase.com/cudalab](http://www.aleacubase.com/cudalab).
You can also folk or clone the source code from [GitHub](https://github.com/soloman817/CUDALab).

The project targets coding [Alea.cuBase](http://www.quantalea.net) examples using F# scripts.
[Alea.cuBase](http://www.quantalea.net) is a language integrated compiler which allows you
to code _CUDA_ kernel with F# and then generate the kernel dynamically and load it into _CUDA_
enabled device.

F# support scripting. We code each examples in this project using F# script. We also use
[FSharp.Formatting](https://github.com/tpetricek/FSharp.Formatting) to do 
[Literate Programming](http://tpetricek.github.io/FSharp.Formatting/demo.html).

All pages are written in markdown or F# scripts. Each F# script is a valid F# code, which can
be run. When building the document output (like HTML), the script will be run and its output
will be automatically added into the bottom of the generated HTML page.

To build the output, you need run the `build.cmd`. But due to a bug you must have Alea.cuBase 
license installed to run the `build.cmd` (This may be fixed in the future).

