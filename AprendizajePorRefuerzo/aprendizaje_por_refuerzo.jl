### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 4f398a75-d3b5-4737-8d2d-d7dd9175467f
using PlutoUI

# ╔═╡ 6a0d9078-fe9e-4263-b4b5-57476200112b
using PlutoTeachingTools

# ╔═╡ 9bce1c4f-050f-48b7-977d-c92dcce4f7eb
TableOfContents(title="Contenidos", depth=1)

# ╔═╡ 86559088-487c-4fce-a226-96dc4c07a9fc
imagenes = "https://belmonte.uji.es/Docencia/IR2130/Teoria/AprendizajePorRefuerzo/Imagenes/";

# ╔═╡ a0e19c16-d03f-11f0-b6de-c7536e08b67c
md"""
# Aprendizaje por Refuerzo

Óscar Belmonte Fernández - IR2130 Aprendizaje Automático

Grado en Inteligencia Robótica - Universitat Jaume I (UJI)
"""

# ╔═╡ 31674476-f661-42a7-af34-26c825da1db4
Resource(
	"https://belmonte.uji.es/imgs/uji.jpg",
	:alt => "Logo UJI",
	:width => 400,
	:style => "display: block; margin: auto;",
)

# ╔═╡ c84b4cd3-a5c3-46b8-a790-d3edd08aa221
md"""
# Introducción

El aprendizaje por refuerzo y el aprendizaje por refuerzo profundo son disciplinas muy amplias. Con el paso del tiempo, los investigadores de estas disciplinas han ido encontrando problemas para solucionar algunos de los problemas del aprendizaje automático.

Como en el caso de las redes neuronales, existen bibliotecas que implementan estos algoritmos. Julia dispone de un conjunto de bibliotecas que implementan los algoritmos que hemos visto, los puedes encontrar [aquí](https://github.com/JuliaPOMDP)
"""

# ╔═╡ 0971106b-5943-4777-9efe-93123e212565
md"""
# Duración de la práctica

A esta práctica le vamos a dedicar una sesión.

!!! danger "Esta práctica no se entrega"
	Esta práctica no tiene entrega.

	El objetivo es que tengas una primera aproximación con el aprendizaje por refuerzo con Julia.
"""

# ╔═╡ 77c12d2f-9567-40e2-b48f-14f4c8d0b6a3
md"""
# Objetivos de aprendizaje

Tener un primer contacto con los algoritmos de aprendizaje por refuerzo.
"""

# ╔═╡ 850130b5-a553-4e65-a29a-0e67ff17eead
md"""
# Metodología

Esta práctica es diferente al resto de prácticas ya que *no vas a realizar ninguna entrega*. En esta práctica vas a explorar las posibilidades que te ofrecen las bibliotecas de Julia para aprendizaje por refuerzo.

En teoría hemos visto el desarrollo de un ejemplo completo para la definición de un entorno, el lago helado. Esta vez vamos a utilizar una biblioteca de modelos disponible en Julia [POMDPModels](https://github.com/JuliaPOMDP/POMDPModels.jl)

Estas son todas la bibliotecas que vamos a utilizar:

```julia
using POMDPs
using POMDPModels
using POMDPTools
using DiscreteValueIteration
using TabularTDLearning
using DeepQLearning
using Flux
using POMDPGifs
using Cairo
using Fontconfig
using StaticArrays
```
"""

# ╔═╡ 4f22c30f-22a4-4583-b6be-e5cdbe2d260c
md"""
# Definición del entorno
"""

# ╔═╡ f5570c7d-b746-4c4c-8196-821a1c65e765
Resource(
	imagenes * "frozen_lake.png",
	:alt => "Lago helado",
	:style => "display: block; margin: auto;",
)

# ╔═╡ b403959f-81ea-4a6e-95cb-dae040793304
md"""
# Definición del entorno

Para crear el lago helado que hemos utilizado en teoría necesitamos indicar:

1. El tamaño del lago.
1. Las recompensas de cada estado.
1. El factor de descuento.

Con todo ellos ya podemos crear el entorno donde va trabajar nuestro agente:

```julia
recompensas = Dict(
                   [1, 1] => -10.0,
                   [4, 2] => -10.0,
                   [2, 3] => -10.0,
                   [4, 3] => -10.0,
                   [4, 1] => 10.0
                  )
entorno = SimpleGridWorld(size = (4, 4), rewards = recompensas, discount = 0.99)
```
"""

# ╔═╡ 0730f9b3-96ca-49e2-9179-84d011b4fc1f
md"""
# Solución con el algoritmo iteración valor

Lo siguiente es definir el solucionador, y resolver el problema.

```julia
solucionador = ValueIterationSolver(; max_iterations=100, belres=1e-3, verbose=true)
politica = solve(solucionador, entorno)
```
"""

# ╔═╡ a4342ae4-14a2-4f0e-b995-a2675288afb8
md"""
# Simulaciones

Una vez que hemos encontrado la política, podemos utilizarla para hacer simulaciones. 

!!! warning "Una simulación puede acabar en fracaso"
	Ten en cuenta que en una simulación podemos acabar en un agujero porque el suelo está helado

```julia
historia = HistoryRecorder(max_steps = 50)
simulate(historia, entorno, politica, SVector(1,4))

```
"""

# ╔═╡ 16fb3b57-946c-405a-8b8d-db1a723c0c60
md"""
Este es el resultado de una simulación:

```shell
(s = [1, 4], a = :down, sp = [2, 4], r = 0.0, info = nothing, t = 1, action_info = nothing)
 (s = [2, 4], a = :right, sp = [3, 4], r = 0.0, info = nothing, t = 2, action_info = nothing)
 (s = [3, 4], a = :down, sp = [3, 3], r = 0.0, info = nothing, t = 3, action_info = nothing)
 (s = [3, 3], a = :down, sp = [3, 2], r = 0.0, info = nothing, t = 4, action_info = nothing)
 (s = [3, 2], a = :down, sp = [3, 3], r = 0.0, info = nothing, t = 5, action_info = nothing)
 (s = [3, 3], a = :down, sp = [3, 2], r = 0.0, info = nothing, t = 6, action_info = nothing)
 (s = [3, 2], a = :down, sp = [3, 1], r = 0.0, info = nothing, t = 7, action_info = nothing)
 (s = [3, 1], a = :right, sp = [4, 1], r = 0.0, info = nothing, t = 8, action_info = nothing)
 (s = [4, 1], a = :up, sp = [-1, -1], r = 10.0, info = nothing, t = 9, action_info = nothing)
```

Hemos tenido suerte y la recompensa al final del episodio es de 10.0.
"""

# ╔═╡ b4d86c5f-f0ff-404a-8114-8b8ab1e3e2b3
md"""
También podemos crear gifs animado con las simulaciones:

```julia
simulador = GifSimulator(filename="simulacion.gif", max_steps=20)
simulate(simulador, entorno, politica, SVector(1,4))
```
"""

# ╔═╡ 02ddfd12-f114-431b-b7c5-8c283ca0654e
Resource(
	"https://belmonte.uji.es/Docencia/IR2130/Practicas/AprendizajePorRefuerzo/Imagenes/simulacion.gif",
	:alt => "Simulación lago helado 4x4",
	:width => 400,
	:style => "display: block; margin: auto;",
)

# ╔═╡ cc3ddc15-f0c0-4ace-99b5-d11e7bd95708
md"""
Puedes notar que, a veces, la acción que se realiza (círculo azul) no es la que el agente selecciona (círculo amarillo con flecha de acción).
"""

# ╔═╡ d0c2dd28-3983-4049-b1f1-199e044d5042
md"""
# Solución con el algoritmo Q-learning

Tenemos a nuestra disposición el algoritmo implementado, usarlo es muy sencillo:

```julia
# Definimos el factor de exploración frente a explotación
politicavoraz = EpsGreedyPolicy(entorno, 0.01)
# Esta vez solucionamos el problema con el algoritmo Q-learning
qsolucionador = QLearningSolver(exploration_policy=politicavoraz, learning_rate=0.1, n_episodes=5000, max_episode_length=100, eval_every=50, n_eval_traj=100)
# Solucionamos el problema obteniendo una política que aproxima a la óptima.
qpolitica = solve(qsolucionador, entorno)
```
"""

# ╔═╡ bb351c3f-eab1-47b1-b09c-e37163bc9ab6
md"""
# Simulaciones

Prueba a hacer simulaciones tal y como las hemos hecho para la política encontrada con el método de iteración valor.
"""

# ╔═╡ 8495e7df-f200-4676-b4d0-d6cf826ff00f
md"""
# Solución con el algoritmo Deep Q-Learning

De nuevo, tenemos implementado el algoritmo, usarlo es muy sencillo. En esta caso, podemos probar a cambiar la red añadiendo más capas, cambiando las funciones de activación, etc.

Primero definimos la red:

```julia
modelo = Chain(
               Dense(2, 32), 
               Dense(32, length(actions(entorno)))
              )
```
Es una red muy sencilla.
"""

# ╔═╡ 5dc488e2-3482-4a3d-9627-2dad6f2b4668
md"""
Como en el caso del algoritmo Q-learning, definimos la política para gestionar la exploración frente a la explotación, el solucionador, y resolvemos:

```julia
exploracion = EpsGreedyPolicy(entorno, LinearDecaySchedule(start=1.0, stop=0.01, steps=10000/2));
dqsolver = DeepQLearningSolver(qnetwork = modelo, max_steps=100000, 
                             exploration_policy = exploracion,
                             learning_rate=0.005,log_freq=500,
                             recurrence=false,double_q=true, dueling=true, prioritized_replay=true)
dqpolitica = solve(dqsolver, entorno)
```

Ya tenemos la nueva política que nos ha devuelto el algoritmo Deep Q-learning.
"""

# ╔═╡ f2ddf5c0-23f4-4308-b1e6-efe19e2b8f54
md"""
# Simulaciones

Prueba a hacer algunas simulaciones. Comprobarás que la política que encuentra el algoritmo Deep Q-Learning con el modelo (red neuronal) que hemos definido no da muy buenos resultados. Modifica la red añadiendo más neuronas, capas, cambian las funciones de activación, y haz varias simulaciones para que veas cómo el algoritmos cada vez va proporcionando mejores soluciones.
"""

# ╔═╡ 78b2a341-8536-46a8-b0f8-7d07ab0c265f
md"""
# Entrega

!!! danger "Esta práctica no se entrega"
	Esta práctica no tiene entrega.

	El objetivo es que tengas una primera aproximación con el aprendizaje por refuerzo con Julia.

"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoTeachingTools = "~0.4.6"
PlutoUI = "~0.7.75"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.2"
manifest_format = "2.0"
project_hash = "bfc763b40872d0365ffe60c17c554c99bfa07c33"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "5b6bb73f555bc753a6153deec3717b8904f5551c"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.3.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4255f0032eafd6451d707a51d5f0248b8a165e4d"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.3+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "44f93c47f9cd6c7e431f2f2091fcba8f01cd7e8f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.10"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.5.20"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoUI"]
git-tree-sha1 = "dacc8be63916b078b592806acd13bb5e5137d7e9"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.4.6"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "db8a06ef983af758d285665a0398703eb5bc1d66"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.75"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "79529b493a44927dd5b13dde1c7ce957c2d049e4"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.6.0"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"
"""

# ╔═╡ Cell order:
# ╟─4f398a75-d3b5-4737-8d2d-d7dd9175467f
# ╟─6a0d9078-fe9e-4263-b4b5-57476200112b
# ╟─9bce1c4f-050f-48b7-977d-c92dcce4f7eb
# ╟─86559088-487c-4fce-a226-96dc4c07a9fc
# ╟─a0e19c16-d03f-11f0-b6de-c7536e08b67c
# ╟─31674476-f661-42a7-af34-26c825da1db4
# ╟─c84b4cd3-a5c3-46b8-a790-d3edd08aa221
# ╟─0971106b-5943-4777-9efe-93123e212565
# ╟─77c12d2f-9567-40e2-b48f-14f4c8d0b6a3
# ╟─850130b5-a553-4e65-a29a-0e67ff17eead
# ╟─4f22c30f-22a4-4583-b6be-e5cdbe2d260c
# ╟─f5570c7d-b746-4c4c-8196-821a1c65e765
# ╟─b403959f-81ea-4a6e-95cb-dae040793304
# ╟─0730f9b3-96ca-49e2-9179-84d011b4fc1f
# ╟─a4342ae4-14a2-4f0e-b995-a2675288afb8
# ╟─16fb3b57-946c-405a-8b8d-db1a723c0c60
# ╟─b4d86c5f-f0ff-404a-8114-8b8ab1e3e2b3
# ╟─02ddfd12-f114-431b-b7c5-8c283ca0654e
# ╟─cc3ddc15-f0c0-4ace-99b5-d11e7bd95708
# ╟─d0c2dd28-3983-4049-b1f1-199e044d5042
# ╟─bb351c3f-eab1-47b1-b09c-e37163bc9ab6
# ╟─8495e7df-f200-4676-b4d0-d6cf826ff00f
# ╟─5dc488e2-3482-4a3d-9627-2dad6f2b4668
# ╟─f2ddf5c0-23f4-4308-b1e6-efe19e2b8f54
# ╟─78b2a341-8536-46a8-b0f8-7d07ab0c265f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
