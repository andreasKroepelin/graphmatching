# Graph Matching using Gradient Flow

This demonstrates the findings of [Zavlanos and Pappas, 2008](https://doi.org/10.1016/j.automatica.2008.04.009) on using *Analog Computing* in the form of gradient flows to solve graph matching problems.

How to run this:
First, clone this repository.
Then, install [Julia](https://julialang.org/downloads/) and start the executable.
Hit `;` to enter the shell-mode.
Navigate to this directory using your usual shell-commands.
Hit backspace to leave shell-mode and then `]` to enter package-mode.
Run
```
(@v1.5) pkg> activate .
(graphmatching) pkg> instantiate
(graphmatching) pkg> precompile
```
Hit backspace again and run
```
julia> using Pluto
julia> Plut.run(notebook = "graph_matching.jl")
```

Follow the displayed instructions and enjoy exploring!
