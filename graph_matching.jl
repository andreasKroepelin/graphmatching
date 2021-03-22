### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 79b40eaa-f6b6-11ea-05a0-d3a166f4fd39
using OrdinaryDiffEq, LightGraphs, GraphPlot, Random, LinearAlgebra, Plots, PlutoUI

# ╔═╡ 2109877c-f6b5-11ea-1941-a7f99c8bab86
md"# Graph Matching using Gradient Flow in ``\cal{O}_n``"

# ╔═╡ ed92164a-8b01-11eb-031d-7997c875255a
md"""
This demonstrates the findings of [Zavlanos and Pappas, 2008](https://doi.org/10.1016/j.automatica.2008.04.009) on using *Analog Computing* in the form of gradient flows to solve graph matching problems.
"""

# ╔═╡ f0fb46aa-f6d8-11ea-2467-1b2fa447a503
md"## Original and permuted graph"

# ╔═╡ 999b3430-8b02-11eb-1076-354db3137ad1
md"""
We start by constructing a simple graph. Just chose one of the built-in examples.
"""

# ╔═╡ 1c300130-fa76-11ea-3a12-47d1dffd179b
md"""**Select example graph:** $(@bind example_graph Select(
	[ "chain5" => "chain of 5"
	, "chain20" => "chain of 20"
	, "tutte" => "tutte"
	, "housex" => "house"
	]))
"""

# ╔═╡ 12ec0c50-8b04-11eb-0c2d-714d2edc1f4b
md"""
And here is a visualisation of that graph.
(The layouting is non-deterministic, so try rerunning the cell if it looks strange.)
"""

# ╔═╡ 40af3fc4-8b04-11eb-2b1c-9b6658331a25
md"""
We consider `G1` to be the "original" graph.
Let's construct another graph, `G2`, by permuting all vertex labels in `G1`:
"""

# ╔═╡ ca129dcc-8b04-11eb-2f89-7f6855899278
md"""
Our **goal** will be to find a permuation of the vertices in `G2` such that we obtain `G1` again.
"""

# ╔═╡ ba799a4e-fa79-11ea-07fb-4fa60e4829d5
md"### Adjacency matrices"

# ╔═╡ 9eb927fe-8b04-11eb-0699-47c6e029292c
md"""
We will use some linear algebra to solve the graph matching.
To do this, we need canonical matrix representation of a graph: its *adjacency matrix*.

Recall that for an undirected graph ``G = (V, E)``, its adjacency matrix ``A \in \{0, 1\}^{|V| \times |V|}`` has entries
```math
a_{ij} = \begin{cases}
	1, & \text{if } \{i, j\} \in E \\
	0, & \text{else.}
\end{cases}
```
"""

# ╔═╡ be035310-8b08-11eb-26af-2d8fe12b1a29
md"""
## Our Strategy
We briefly restate the strategy that Zavlanos and Pappas suggest:

We want to permute the vertices in `G2`, which is equivalent to permuting the rows and columns in `A2`.
More specifically, we want to find a permuatation matrix ``P`` such that ``A_1`` and ``P^\top A_2 P`` (or equivalently ``P A_1`` and ``A_2 P``) are as similar as possible.
To quantify the approximation error, we pick the squared *Frobenius*-norm as our notion of similarity, which just means that we sum the elementwise squared differences of both matrices:
"""

# ╔═╡ 3a7dd4d2-8b0f-11eb-0658-f3d3779b33f3
md"### Minimise approximation error"

# ╔═╡ c748b142-8b0b-11eb-0199-c5924c5d19cf
md"""
The authors then explain that the permutation matrices are exactly the orthogonal matrices with non-negative entries.
We exploit that by first relaxing our problem to ``P`` being any orthogonal matrix and performing a **gradient flow**.

This is the ODE problem and its solution that describes the gradient flow:
"""

# ╔═╡ 262ef806-f6c4-11ea-1b96-8776b89f8071
function min_approx_err!(dP, P, params, t)
	A1, A2 = params
	dP .= P * (P' * A2 * P * A1 - A1 * P' * A2 * P)
end

# ╔═╡ 9d320016-8b0d-11eb-0d83-45212b00c532
md"Define for how long we want to follow the gradient flow:"

# ╔═╡ 6092784c-f741-11ea-1a0b-3b40c5a88a67
t1 = 20.0

# ╔═╡ 1a9da0ce-f6da-11ea-24a1-398e86eea280
md"""
### Penalising negative entries in ``P``

We now need to make the orthogonal matrix into a permutation matrix, which means we have to get rid of the negative entries.

This function assesses how negative the entries in ``P`` are:
"""

# ╔═╡ ca89c966-f6cf-11ea-05d8-717423c34f3c
negativity(P) = 2/3 * tr(P' * (P .- P .* P))

# ╔═╡ f161ae44-8b0f-11eb-07fc-5364c9e5aa55
md"""
And here we define the corresponding gradient flow:
"""

# ╔═╡ 222dbe04-f6d0-11ea-2997-b5d642fddf4c
function penalise_negative!(dP, P, params, t)
	PP = P .* P
	dP .= P * (P' * PP .- PP' * P)
end

# ╔═╡ 5d6b088e-8b0f-11eb-26f6-dfa9cacb669e
md"""
It turns out that it makes sense to actually combine the gradient flows for the actual objective and for penalising negative entries with some weighting factor `k`.
"""

# ╔═╡ cb658266-fa79-11ea-0db1-7567dea3d069
k = 0.5

# ╔═╡ 490ce0c0-f72f-11ea-1466-3b1aeb47fc11
function combined_gradient!(dP, P, params, t)
	A1, A2, k = params
	dP1 = similar(dP)
	dP2 = similar(dP)
	min_approx_err!(dP1, P, (A1, A2), t)
	penalise_negative!(dP2, P, nothing, t)
	dP .= (1-k) * dP1 .+ k * dP2
end

# ╔═╡ 6767b2d2-8b10-11eb-116c-1972bef114fe
md"""
And this is the overall ODE problem describing this second gradient flow, which starts at the last ``P`` of the first system and evolves until `t2`.
"""

# ╔═╡ cabbd10e-fa79-11ea-30f2-bd1c70613a3c
t2 = 60.0

# ╔═╡ e63643ac-f739-11ea-32ac-d124ec648ff8
md"""
We can summarise the strategy:

**Minimise approximation error until time $t1, then additionally make entries in ``P`` non-negative until time $t2 with $(round(Int, 100k)) % importance.**
"""

# ╔═╡ ba3571ac-8b10-11eb-3654-f3861db16d26
md"""
## Results

This combines both parts of the gradient flow into one time-dependent function:
"""

# ╔═╡ 05f036a2-fa7c-11ea-1902-7592154ca34d
md"### Plotting"

# ╔═╡ a730b14e-f74e-11ea-3ec9-ed8b3a479bdc
trange = 0 : 1.0 : t2;

# ╔═╡ 02cf1af8-8b11-11eb-00a1-2b2e298b4a85
md"""
This is the final solution for ``P`` that the algorithm came up with:
"""

# ╔═╡ 17f5d944-8b11-11eb-0f62-c5e233214748
md"""
Sanity check, that the final solution is still orthogonal:
"""

# ╔═╡ 369c1c8a-f6da-11ea-2069-3fde5a57b953
md"""
## Reconstructed graph

Finally, we can perfom the permutation and find the reconstructed adjacency matrix:
"""

# ╔═╡ 59b29f46-8b11-11eb-2b3c-277d36a0debb
md"""
And this is the corresponding graph.
Does it look at least somewhat like `G1`?
I hope so!
Try playing with the `k` parameter to see how that affects the solution quality.
"""

# ╔═╡ 64e538b2-f6bf-11ea-375e-a5f3e77d6cd0
function permute_vertices(g::AbstractGraph)
	perm_g = SimpleGraph(nv(g))
	perm_v = shuffle(vertices(g))
	
	for e in edges(g)
		perm_s = perm_v[src(e)]
		perm_d = perm_v[dst(e)]
		add_edge!(perm_g, perm_s, perm_d)
	end
	
	perm_g
end

# ╔═╡ 12c0e9ee-fa77-11ea-3e61-ed34d1ea848e
function construct_example(id)
	if id == "tutte"
		smallgraph(:tutte)
	elseif id == "housex"
		smallgraph(:housex)
	elseif id == "chain5"
		path_graph(5)
	elseif id == "chain20"
		path_graph(20)
	else
		path_graph(1)
	end
end

# ╔═╡ c548c0b2-f6bc-11ea-1ba0-f3de59ef9987
G1 = construct_example(example_graph)

# ╔═╡ a9ac43ac-f6c2-11ea-2114-338b468c054f
gplot(G1, nodelabel=collect(vertices(G1)))

# ╔═╡ 14eeb4fe-f6c0-11ea-1211-972d5d4f6a4b
G2 = permute_vertices(G1)

# ╔═╡ 78bb17c8-f6c2-11ea-04df-69c5d723ec3c
gplot(G2, nodelabel=collect(vertices(G2)))

# ╔═╡ d8cbb096-f6c2-11ea-1f32-ed95b80b4d8f
A2 = adjacency_matrix(G2);

# ╔═╡ cc0052ea-f6c2-11ea-2710-e175f7ec3f8a
A1 = adjacency_matrix(G1);

# ╔═╡ 637607f8-f6cc-11ea-2a32-a15f4d93c7e7
approx_err(P) = .5 * sum((P*A1 - A2*P) .^ 2)

# ╔═╡ afba6f40-f6c6-11ea-20a3-e16e48da9005
function rand_SO(s::Tuple{Int,Int})
	Q, _ = rand(s...) |> qr
	Q = Matrix(Q)
	if det(Q) < 0
		Q[1,:] .*= -1
	end
	Q
end

# ╔═╡ f4fd59b4-f6c6-11ea-165b-eb798b8fbff5
min_approx_err_solution = let
	P0 = rand_SO(size(A1))
	t0 = 0.0
	min_approx_err_problem = ODEProblem(min_approx_err!, P0, (t0, t1), (A1, A2))
	
	solve(min_approx_err_problem, Tsit5())
end;

# ╔═╡ 19739754-fa7a-11ea-2b0e-037d4266a5ae
P_min_approx_err = min_approx_err_solution(t1);

# ╔═╡ 7519397a-f6d0-11ea-19ad-f5aaa7f8f228
non_negative_solution = let
	non_negative_problem = ODEProblem(combined_gradient!, P_min_approx_err, (t1, t2), (A1, A2, k))
	solve(non_negative_problem, Tsit5())
end;

# ╔═╡ f6580a7e-f6d1-11ea-2200-f9c80bc69b31
P(t) = if t < t1
		min_approx_err_solution(t)
	else
		non_negative_solution(t)
	end

# ╔═╡ 3358b806-f6d2-11ea-203e-d53ae160ffd6
anim = map(trange) do t
	l = heatmap(P(t) * A1 - A2 * P(t) .|> abs,
		clim=(0,2),
		title="t = $(round(Int, t))",
		xlabel="elem. wise deviation")
	r = heatmap(P(t),
		clim=(-1,1),
		xlabel="permutation matrix")
	plot(l, r, layout=(1,2), size=(600,300), aspect_ratio=:equal)
end;

# ╔═╡ ce19a088-fa7b-11ea-30b3-7b80bfbbdf1f
@bind t_anim Slider(eachindex(anim))

# ╔═╡ ec1ba20c-fa7b-11ea-1238-41e99f6ded6b
anim[t_anim]

# ╔═╡ 507ccbd4-f6d2-11ea-3a8d-1311265cdc54
begin
	plot(trange,
		[[approx_err(P(t)) for t in trange],
		[negativity(P(t)) for t in trange]],
		label=["approximation error"  "negativity of entries"],
		xlabel="time")
	vline!([t1]; linestyle=:dash, label="switching systems")
	vline!([trange[t_anim]]; linestyle=:solid, label="currently displayed")
end

# ╔═╡ 422d4c82-f6d4-11ea-3e87-d7f548a73eb7
final_P = P(t2);

# ╔═╡ e527660c-fc60-11ea-309e-c7372eea3e23
round.(final_P, digits=3)

# ╔═╡ 41f9aea8-f6d5-11ea-0330-af1c77272674
round.(final_P' * final_P, digits=3)

# ╔═╡ a184f726-f6d8-11ea-2c93-8fb47ac719a3
A3 = round.(Int, final_P' * A2 * final_P)

# ╔═╡ 2735a812-f6d8-11ea-16c4-dbbc3b090adb
G3 = SimpleGraph(A3)

# ╔═╡ 7649f264-f6d8-11ea-1712-5dac97376cdc
gplot(G3, nodelabel=collect(vertices(G3)))

# ╔═╡ 97c742fc-f6d5-11ea-1b58-6be1318afa25
begin
	plot(trange,
		[minimum(P(t)) for t in trange],
		ylim=(-1,0.1),
		label="min. entry in P",
		xlabel="time")
	vline!([t1]; linestyle=:dash, label="switching systems", legend=:bottomright)
end

# ╔═╡ Cell order:
# ╟─2109877c-f6b5-11ea-1941-a7f99c8bab86
# ╟─ed92164a-8b01-11eb-031d-7997c875255a
# ╟─f0fb46aa-f6d8-11ea-2467-1b2fa447a503
# ╟─999b3430-8b02-11eb-1076-354db3137ad1
# ╟─1c300130-fa76-11ea-3a12-47d1dffd179b
# ╠═c548c0b2-f6bc-11ea-1ba0-f3de59ef9987
# ╟─12ec0c50-8b04-11eb-0c2d-714d2edc1f4b
# ╠═a9ac43ac-f6c2-11ea-2114-338b468c054f
# ╟─40af3fc4-8b04-11eb-2b1c-9b6658331a25
# ╠═14eeb4fe-f6c0-11ea-1211-972d5d4f6a4b
# ╠═78bb17c8-f6c2-11ea-04df-69c5d723ec3c
# ╟─ca129dcc-8b04-11eb-2f89-7f6855899278
# ╟─ba799a4e-fa79-11ea-07fb-4fa60e4829d5
# ╟─9eb927fe-8b04-11eb-0699-47c6e029292c
# ╠═cc0052ea-f6c2-11ea-2710-e175f7ec3f8a
# ╠═d8cbb096-f6c2-11ea-1f32-ed95b80b4d8f
# ╟─be035310-8b08-11eb-26af-2d8fe12b1a29
# ╠═637607f8-f6cc-11ea-2a32-a15f4d93c7e7
# ╟─3a7dd4d2-8b0f-11eb-0658-f3d3779b33f3
# ╟─c748b142-8b0b-11eb-0199-c5924c5d19cf
# ╠═262ef806-f6c4-11ea-1b96-8776b89f8071
# ╟─9d320016-8b0d-11eb-0d83-45212b00c532
# ╠═6092784c-f741-11ea-1a0b-3b40c5a88a67
# ╠═f4fd59b4-f6c6-11ea-165b-eb798b8fbff5
# ╠═19739754-fa7a-11ea-2b0e-037d4266a5ae
# ╟─1a9da0ce-f6da-11ea-24a1-398e86eea280
# ╠═ca89c966-f6cf-11ea-05d8-717423c34f3c
# ╟─f161ae44-8b0f-11eb-07fc-5364c9e5aa55
# ╠═222dbe04-f6d0-11ea-2997-b5d642fddf4c
# ╟─5d6b088e-8b0f-11eb-26f6-dfa9cacb669e
# ╠═cb658266-fa79-11ea-0db1-7567dea3d069
# ╠═490ce0c0-f72f-11ea-1466-3b1aeb47fc11
# ╟─6767b2d2-8b10-11eb-116c-1972bef114fe
# ╠═cabbd10e-fa79-11ea-30f2-bd1c70613a3c
# ╠═7519397a-f6d0-11ea-19ad-f5aaa7f8f228
# ╟─e63643ac-f739-11ea-32ac-d124ec648ff8
# ╟─ba3571ac-8b10-11eb-3654-f3861db16d26
# ╠═f6580a7e-f6d1-11ea-2200-f9c80bc69b31
# ╟─05f036a2-fa7c-11ea-1902-7592154ca34d
# ╠═a730b14e-f74e-11ea-3ec9-ed8b3a479bdc
# ╟─3358b806-f6d2-11ea-203e-d53ae160ffd6
# ╟─ce19a088-fa7b-11ea-30b3-7b80bfbbdf1f
# ╟─ec1ba20c-fa7b-11ea-1238-41e99f6ded6b
# ╟─507ccbd4-f6d2-11ea-3a8d-1311265cdc54
# ╟─02cf1af8-8b11-11eb-00a1-2b2e298b4a85
# ╠═422d4c82-f6d4-11ea-3e87-d7f548a73eb7
# ╠═e527660c-fc60-11ea-309e-c7372eea3e23
# ╟─17f5d944-8b11-11eb-0f62-c5e233214748
# ╠═41f9aea8-f6d5-11ea-0330-af1c77272674
# ╟─97c742fc-f6d5-11ea-1b58-6be1318afa25
# ╟─369c1c8a-f6da-11ea-2069-3fde5a57b953
# ╠═a184f726-f6d8-11ea-2c93-8fb47ac719a3
# ╟─59b29f46-8b11-11eb-2b3c-277d36a0debb
# ╠═2735a812-f6d8-11ea-16c4-dbbc3b090adb
# ╠═7649f264-f6d8-11ea-1712-5dac97376cdc
# ╟─64e538b2-f6bf-11ea-375e-a5f3e77d6cd0
# ╟─12c0e9ee-fa77-11ea-3e61-ed34d1ea848e
# ╟─afba6f40-f6c6-11ea-20a3-e16e48da9005
# ╠═79b40eaa-f6b6-11ea-05a0-d3a166f4fd39
