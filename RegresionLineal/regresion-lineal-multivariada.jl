### A Pluto.jl notebook ###
# v0.20.19

using Markdown
using InteractiveUtils

# ╔═╡ 0bd06eba-87db-11f0-2fd9-a1c82ad71b09
using PlutoUI # Interactividad con Pluto

# ╔═╡ 8814a209-1c19-467a-933d-10469506c51c
TableOfContents(title="Contenidos", depth=1)

# ╔═╡ 78c37d96-3c62-4159-97f0-48f81b5b6019
md"""
# Regresión lineal multivariada

Óscar Belmonte Fernández - IR2130 Aprendizaje Automático

Grado en Inteligencia Robótica - Universitat Jaume I (UJI)
"""

# ╔═╡ 67e10dea-e182-49b5-ab5d-77c849d26872
Resource(
	"https://belmonte.uji.es/imgs/uji.jpg",
	:alt => "Logo UJI",
	:width => 400,
	:style => "display: block; margin: auto;",
)

# ╔═╡ be6e3754-b43f-42d5-9015-6bdb9bdfcb81
md"""
# Introducción

En esta práctica vas a crear un modelo lineal con múltiples variables
independientes para estimar una única variable dependiente.

El modelo deberá dar una estimación de la edad de cierta variedad de caracoles de mar, a partir de sus características. Los datos los puedes encontrar [aquí](https://raw.githubusercontent.com/AprendizajeAutomaticoUJI/DataSets/refs/heads/master/Abalone/abalone.data), junto con una [descripción](https://raw.githubusercontent.com/AprendizajeAutomaticoUJI/DataSets/refs/heads/master/Abalone/abalone.names) de cada una de las características de las muestras.

Vas a crear un libro de notas de Pluto para desarrollar la práctica. Este libro lo debes subir a aulavirtual para su evaluación.
"""

# ╔═╡ 3393a7cd-f802-4363-bfae-f53536dd8c7a
md"""
# Duración de la práctica

A esta práctica le vamos a dedicar dos sesiones de prácticas.
"""

# ╔═╡ b1b6fdc6-eb9c-4d6d-a77e-c689fbe9c2b0
md"""
# Objetivos de aprendizaje

1. Examinar un conjunto de datos multivariado y valorar su utilidad.
1. Construir un modelo de regresión lineal multivariado.
1. Adaptar el modelo para mejorar su rendimiento.
1. Plantear posibles mejoras del modelo.
1. Emplear el esquema de **Proyectos de Aprendizaje Automático** como guía de desarrollo.
"""

# ╔═╡ fa751513-2fc7-49aa-a581-4dc2071041ad
md"""
# Metodología

Vas a seguir una adaptación de los pasos que has visto en la presentación de 
teoría **Proyectos de Aprendizaje Automático**.

1. Definir el problema y tener una imagen del conjunto.
1. Obtener los datos.
1. Explorar los datos para conocerlos mejor.
1. Preparar los datos para que muestren los patrones.
1. Crear un primera versión del algoritmo.
1. Ajustar el modelo para obtener una solución.
1. Presentar la solución.
1. Crítica del trabajo y posibles mejoras.
"""

# ╔═╡ 87095e1f-dd7d-4762-a1bc-bd4d25a04b17
md"""
# Objetivo

Crear un modelo de regresión lineal multivariado para estimar el valor de una 
variable dependiente a partir de un conjunto de otras variables.
"""

# ╔═╡ 0398e1bc-aff1-4e63-886b-04d344d7aedf
md"""
# Tareas a realizar

Vas a seguir una adaptación del esquema que se presentó en el tema de
**Proyectos de Aprendizaje Automático**.
"""

# ╔═╡ cb208fa1-fe6a-4b95-86b9-54fbf3a073b6
md"""
## Definir el problema y tener un imagen del conjunto

El problema en este caso es calcular la edad de un caracol de mar a partir 
del resto de variables del conjunto de datos.

La edad de uno de estos caracoles se calcula como el número de anillos + 1.5.
"""

# ╔═╡ 22438c9e-c473-489a-93fb-68d527a39fa9
md"""
## Obtener los datos

Los datos los puedes descargar desde el repositorio 
[Datos Abalone](https://raw.githubusercontent.com/AprendizajeAutomaticoUJI/DataSets/refs/heads/master/Abalone/abalone.data).

[Aquí](https://raw.githubusercontent.com/AprendizajeAutomaticoUJI/DataSets/refs/heads/master/Abalone/abalone.names) tienes una descripción de los datos.

Es interesante tener un primer contacto con los datos utilizando el método **describe(df)**.
"""

# ╔═╡ e8269a48-36e1-4aa3-a279-ef14497349a5
md"""
## Explorar los datos para conocerlos mejor

En este caso cada una de las muestras tiene más de dos características, por lo 
que no puedes representar todos los datos en un único gráfico. Sin embargo, sí 
que puede representar gráficas con pares de características para tener una 
primera idea de sus dependencias, para ello es muy útil el método **pairplot** del paquete **PairPlots**:

```julia
using PairPlots

pairplot(df) # df es un DataFrame
```

¿Cuál es la correlación del resto de variables con el número de anillos?
"""

# ╔═╡ a16c7d5c-65fd-4939-b908-0a232bdee53a
md"""
## Preparar lo datos para que muestren los patrones

Este conjunto de datos ya está *limpio*, no hay datos faltantes, afortunadamente no tenemos que preparar los datos.

Utiliza el paquete [**MLJ**](https://juliaml.ai/) para dividir el conjunto inicial de los datos en dos subconjuntos, el conjunto de datos de entrenamiento (80% de los datos), y el conjunto de datos de prueba (20% restante de los datos). Recuerda especificar el origen de generación de los datos aleatorios para que tus experimentos sean reproducibles.

```julia
using MLJ

entrenamiento, prueba = partition(df, 0.8, rng = 69)
```

"""

# ╔═╡ 4f5228a1-226f-4d04-886c-757f87c385eb
md"""
## Crear una primera versión del modelo

Vamos utilizar el paquete [**MLJ**](https://juliaml.ai/) para construir el modelo y analizar los resultados.

Esta es una pequeña receta de cómo trabajar con MLJ:

```julia
using MLJ # No hace falta esta línea, es para que quede más claro el mecanismo.
using GLM # El modelo que vamos a utilizar, LinearRegressor, está en este paquete.
using MLJGLMInterface # Nos hace falta para «envolver» el modelo LinearRegressor y 					  # que MLJ pueda trabajar con él.

LinearRegresor = @load LinearRegressor pkg=GLM # Cargamos el modelo.
regresor = LinearRegresor() # Creamos una instancia.
maquina = machine(regresor, X, y) |> fit! # Creamos la máquina y la entrenamos.
predict_mean(maquina, Xprueba) # Hacemos predicciones.
```

Elige una medida de error para poder comparar las predicciones del modelo.

¿Los residuos siguen una distribución normal? Representa la distribución de los 
residuos y el diagrama qq-plot.

```julia
using StatsPlots

qqnorm(residuos)
```
"""

# ╔═╡ b8146f5c-758d-48da-a402-eaa2c1da7fda
md"""
## Ajustar el modelo para mejorar la solución

¿Cómo puedes mejorar los resultados del modelo?

Una pista, utiliza la variable categórica *Sex* para ajustar los datos por sexo.
"""

# ╔═╡ 7150628a-03ae-4413-bec9-8d89c5912879
md"""
## Presentar la solución

Mejora tu libro de notas para que sea descriptivo del trabajo que has realizado. 
Presta especial atención a la sección de conclusiones.
"""

# ╔═╡ 5e42450f-9eea-49a0-bdfd-b3ccfdef14c9
md"""
## Críticas al trabajo y posibles mejoras

¿Qué crees que se puede mejorar del trabajo que has hecho?

Haz un listado de posibles mejoras, y ordénalas por importancia de mayor a menor.
"""

# ╔═╡ 6a108dc3-c902-4701-bbc2-d36ca5bcb203
md"""
# Entrega

El trabajo que debes entregar para su corrección es el libro de notas 
(fichero con extensión jl).

Súbelo a aulavirtual. No es necesario que lo suban todos los miembros del 
equipo, basta con que lo suba uno de vosotros.

Recuerda escribir al inicio del fichero el nombre de los miembros de la pareja.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.68"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.7"
manifest_format = "2.0"
project_hash = "bf4a620d59312f26669e73defd791ec8e864a597"

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

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

    [deps.ColorTypes.weakdeps]
    StyledStrings = "f489334b-da3d-4c2e-b8f0-e476e12c162b"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

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
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "ec9e63bd098c50e4ad28e7cb95ca7a4860603298"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.68"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

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
git-tree-sha1 = "372b90fe551c019541fafc6ff034199dc19c8436"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.12"

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
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─0bd06eba-87db-11f0-2fd9-a1c82ad71b09
# ╟─8814a209-1c19-467a-933d-10469506c51c
# ╟─78c37d96-3c62-4159-97f0-48f81b5b6019
# ╟─67e10dea-e182-49b5-ab5d-77c849d26872
# ╟─be6e3754-b43f-42d5-9015-6bdb9bdfcb81
# ╟─3393a7cd-f802-4363-bfae-f53536dd8c7a
# ╟─b1b6fdc6-eb9c-4d6d-a77e-c689fbe9c2b0
# ╟─fa751513-2fc7-49aa-a581-4dc2071041ad
# ╟─87095e1f-dd7d-4762-a1bc-bd4d25a04b17
# ╟─0398e1bc-aff1-4e63-886b-04d344d7aedf
# ╟─cb208fa1-fe6a-4b95-86b9-54fbf3a073b6
# ╟─22438c9e-c473-489a-93fb-68d527a39fa9
# ╟─e8269a48-36e1-4aa3-a279-ef14497349a5
# ╟─a16c7d5c-65fd-4939-b908-0a232bdee53a
# ╟─4f5228a1-226f-4d04-886c-757f87c385eb
# ╟─b8146f5c-758d-48da-a402-eaa2c1da7fda
# ╟─7150628a-03ae-4413-bec9-8d89c5912879
# ╟─5e42450f-9eea-49a0-bdfd-b3ccfdef14c9
# ╟─6a108dc3-c902-4701-bbc2-d36ca5bcb203
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
