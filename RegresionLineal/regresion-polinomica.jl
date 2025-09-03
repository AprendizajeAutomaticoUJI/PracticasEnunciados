### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# ╔═╡ bfa26380-88a3-11f0-193c-f10c4a53c567
using PlutoUI # Interactividad con Pluto

# ╔═╡ aad6c742-ff75-48f7-aa06-36a1d9382618
TableOfContents(title="Contenidos", depth=1)

# ╔═╡ 5fdf9a7d-c1ee-4a82-a9a3-05a7470ec662
md"""
# Introducción

En esta práctica vas a crear un modelo de regresión polinomial para un
conjunto de datos de dos variables, una variable independiente y una variable 
dependiente.

Los datos proporcionan información sobre la energía generado por una planta eólica y la velocidad del viento. Tu modelo deberá estimar la energía producida por la planta eólica a partir del dato de velocidad del viento. 

Los datos los puedes descargar desde [aquí](https://raw.githubusercontent.com/AprendizajeAutomaticoUJI/DataSets/master/powerproduction.csv).

Vas a crear un libro de notas de Pluto para desarrollar la práctica. Este 
libro lo debes subir a aulavirtual para su evaluación.
"""

# ╔═╡ 1540974b-4987-44bc-b638-63aab0c3062b
md"""
# Duración de la práctica

A esta práctica le vamos a dedicar una sesión.
"""

# ╔═╡ 8f411dd8-96b6-4d60-9f43-9c7ef4bdd569
md"""
# Objetivos de aprendizaje

1. Examinar un conjunto de datos de dos variables y valorar su utilidad.
1. Construir un modelo de regresión polinomial.
1. Adaptar el modelo para mejorar su rendimiento.
1. Plantear posibles mejoras del modelo.
1. Emplear el esquema de **Proyectos de Aprendizaje Automático** como guía de 
desarrollo.
"""

# ╔═╡ 7f390ed7-a596-47b0-96a7-318124323a9a
md"""
# Metodología

Vas a seguir una adaptación de los pasos que has visto en la presentación de teoría **Proyectos de Aprendizaje Automático**.

1. Definir el problema y tener una imagen del conjunto.
1. Obtener los datos.
1. Explorar los datos para conocerlos mejor.
1. Preparar los datos para que muestren los patrones.
1. Crear una primera versión del algoritmo.
1. Ajustar el modelo para obtener una solución.
1. Presentar la solución.
1. Críticas al trabajo y posibles mejoras.
"""

# ╔═╡ 661cb223-9353-4c25-ba80-05fc4ea2524a
md"""
# Objetivo

Crear un modelo de regresión polinómica que ajuste los datos un conjunto con medidas de generación de energía de un aerogenerador y la velocidad del viento.

A partir de este modelo podrás estimar la energía producida por el aerogenerador para la velocidad del viento actual.
"""

# ╔═╡ 74e9b1e1-951b-4ded-9c5f-f4ca2c9af333
md"""
# Tareas a realizar

Vas a seguir una adaptación del esquema que se presentó en el tema de **Proyectos de Aprendizaje Automático**.
"""

# ╔═╡ 2608ec4d-0cfe-4b3f-9fb0-6180d1a142ea
md"""
## Definir el problema y tener un imagen del conjunto

Debes redactar, con tus propias palabras en qué consiste el problema que se plantea.
"""

# ╔═╡ a0151c2e-0aa5-4e0a-93d9-ff4c58f220f7
md"""
## Obtener los datos

Los datos los puedes descargar desde el  [aquí](https://raw.githubusercontent.com/AprendizajeAutomaticoUJI/DataSets/master/powerproduction.csv).

El conjunto de datos está formado por una serie de medidas, cada una de ellas presenta la energía generada por un aerogenerador (en kilowatios), para una determinada velocidad del viento (m./s.).
"""

# ╔═╡ 3a1ec0ba-db78-4130-a6f0-29c7677701c4
md"""
## Explorar los datos para conocerlos mejor

Como los datos son bidimensionales, los podemos representar de manera completa en un gráfico 2D. Fíjate en que hay algunos datos *anómalos*, para los que la potencia producida es 0 aún cuando sopla el viento.

Visualiza las distribuciones de ambas variables. ¿Reconoces alguna de las distribuciones?

¿Existe correlación entre las dos características?
"""

# ╔═╡ 826d739c-1663-4e4e-8fa6-24a784c728fe
md"""
## Preparar lo datos para que muestren los patrones

Elimina los datos anómalos y vuelve a representarlos en un gráfico 2D.

¿Qué patrón pueden estar siguiendo los datos? Por ejemplo ¿es adecuado describir la relación entre ellos de manera lineal?
"""

# ╔═╡ e726f9e3-c606-434e-a783-923f7aab4ee0
md"""
## Crear una primera versión del modelo

Ahora que ya tienes una primera hipótesis de cómo crear un modelo, utiliza el paquete [**GLM**]() para implementarlo.

Divide el conjunto inicial de los datos en dos subconjuntos, el conjunto de datos de entrenamiento (80% de los datos), y el conjunto de datos de prueba (20% restante de los datos). Recuerda especificar el origen de generación de los datos aleatorios para que tus experimentos sean reproducibles.

Elige una medida de error para poder comparar los resultados entre varios modelos.

¿Los residuos siguen una distribución normal? Representa la distribución de los residuos y el diagrama qq-plot.
"""

# ╔═╡ 046f1bfa-7b8c-42ba-8c2d-c8cbf237ad14
md"""
## Ajustar el modelo para obtener una solución

Haz una serie de experimentos variando el grado del polinomio a ajustar. Registra el error para cada uno de ellos.

¿Qué grado del polinomio da el error más pequeño? ¿Es razonable utilizar ese grado del polinomio? ¿Por qué?

Prueba a hacer regresiones utilizando regularización (Ridge y Lasso). ¿Mejoran los resultados? ¿Qué conclusiones puedes extraer al comparar los modelos regularizados con la regresión polinómica sin regularización?
"""

# ╔═╡ 936a6fcb-a87b-4089-866d-927290a1dd38
md"""
## Presentar la solución

Mejora tu libro de notas para que sea descriptivo del trabajo que has realizado. Presta especial atención a la sección de conclusiones.
"""

# ╔═╡ 21d0d337-f5a5-4cef-878b-f760757e4ed5
md"""
## Critica del trabajo y posibles mejoras

¿Qué crees que se puede mejorar del trabajo que has hecho? Haz un listado de posibles mejoras, y ordénalas por importancia de mayor a menor.
"""

# ╔═╡ 09331abb-3273-4784-909a-94947cdc5f70
md"""
# Entrega

El trabajo que debes entregar para su corrección es el libro de notas (fichero con extensión jl).

Súbelo a aulavirtual. No es necesario que los suban todos los miembros del equipo, basta con que lo suba uno de vosotros.
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

julia_version = "1.11.6"
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
# ╠═bfa26380-88a3-11f0-193c-f10c4a53c567
# ╠═aad6c742-ff75-48f7-aa06-36a1d9382618
# ╠═5fdf9a7d-c1ee-4a82-a9a3-05a7470ec662
# ╠═1540974b-4987-44bc-b638-63aab0c3062b
# ╠═8f411dd8-96b6-4d60-9f43-9c7ef4bdd569
# ╠═7f390ed7-a596-47b0-96a7-318124323a9a
# ╠═661cb223-9353-4c25-ba80-05fc4ea2524a
# ╠═74e9b1e1-951b-4ded-9c5f-f4ca2c9af333
# ╠═2608ec4d-0cfe-4b3f-9fb0-6180d1a142ea
# ╠═a0151c2e-0aa5-4e0a-93d9-ff4c58f220f7
# ╠═3a1ec0ba-db78-4130-a6f0-29c7677701c4
# ╠═826d739c-1663-4e4e-8fa6-24a784c728fe
# ╠═e726f9e3-c606-434e-a783-923f7aab4ee0
# ╠═046f1bfa-7b8c-42ba-8c2d-c8cbf237ad14
# ╠═936a6fcb-a87b-4089-866d-927290a1dd38
# ╠═21d0d337-f5a5-4cef-878b-f760757e4ed5
# ╠═09331abb-3273-4784-909a-94947cdc5f70
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
