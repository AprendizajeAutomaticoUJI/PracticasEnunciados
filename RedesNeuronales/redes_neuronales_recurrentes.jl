### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 3aa149be-9d44-11f0-17aa-eb647770c269
using PlutoUI

# ╔═╡ 54cd69eb-b10b-4dbe-86d7-18493250efac
TableOfContents(title="Contenidos", depth=1)

# ╔═╡ c4c50b45-e3c6-4ed8-9d98-596f0d30f419
md"""
# Redes neuronales recurrentes

Óscar Belmonte Fernández - IR2130 Aprendizaje Automático

Grado en Inteligencia Robótica - Universitat Jaume I (UJI)
"""

# ╔═╡ a1758fe4-7afe-42ff-835c-4ff710b145eb
Resource(
	"https://belmonte.uji.es/imgs/uji.jpg",
	:alt => "Logo UJI",
	:width => 400,
	:style => "display: block; margin: auto;",
)

# ╔═╡ c03e3d69-6460-4608-8a1e-fe64e40146ec
md"""
# Introducción

Las redes neuronales recurrentes, y en particular las LSTM, *aprenden* las correlaciones entre los datos de una serie temporal, y son capaces de hacer predicciones.

Las series de datos temporales suelen presentar una alta correlación en sus datos, además de otros posibles fenómenos como una tendencia a lo largo del tiempo, y sobre esa tendencia una estacionalidad (variación periódica).

En esta práctica vas a implementar una red neuronal recurrente y una red LSTM para predecir la variación del [Euribor](https://es.wikipedia.org/wiki/Eur%C3%ADbor).

"""

# ╔═╡ 40722903-fe77-41cd-8d4f-6ba87c46dec9
md"""
# Duración de la práctica
A esta práctica le vamos a dedicar una sesión.
"""

# ╔═╡ 565b929f-b551-47ba-b05b-10494540fbec
md"""
# Objetivos de aprendizaje

* Examinar los datos de una serie temporal para detectar su tendencia, estacionalidad y autocorrelación.
* Crear una red neuronal recurrente para predecir la evolución del Euribor.
* Crear una red neuronal LSTM para predecir la evolución del Euribor.
* Valorar el rendimiento de ambas implementaciones.
"""

# ╔═╡ f7bb6033-92c4-4dd1-8a28-e7449ea6b036
md"""
# Metodología
Vas a seguir una adaptación de los pasos que has visto en la presentación de teoría **Proyectos de Aprendizaje Automático**.

1. Definir el problema y tener una imagen del conjunto.
1. Obtener los datos.
1. Explorar los datos para conocerlos mejor.
1. Preparar los datos para que muestren los patrones.
1. Crear un primera versión del modelo. 
1. Ajustar el modelo para obtener una solución.
1. Presentar la solución.
1. Crítica del trabajo y posibles mejoras.
"""

# ╔═╡ e2f1cead-58cb-49e0-8908-80e0477a2441
md"""
# Objetivo

Como primera tarea, vas a analizar la autocorrelación de los datos, si presentan algún tipo de tendencia y/o estacionalidad. Una vez que conozcas los datos vas a pasar a implementar un par de modelos que te permitan predecir el valor del Euribor.

Vas a implementar un modelo de red recurrente y otro de red LSTM para predecir la evolución del Euribor en los tres últimos años de la serie temporal (36 meses). Para ello vas 
"""

# ╔═╡ 2a622058-e74a-4a03-8299-d0d71a180871
md"""
# Tareas a realizar

Vas a seguir una adaptación del esquema que se presentó en el tema de **Proyectos de Aprendizaje Automático**.
"""

# ╔═╡ 0f9d6c20-5ba9-4da4-8225-f7e6a1d33cca
md"""
## Definir el problema y tener una imagen del conjunto

Describe, con tus propias palabras, cuál es la problema que se pretende resolver y cuál es su alcance.
"""

# ╔═╡ b6ef3b13-54fc-4f81-a544-7ae49fca2d2b
md"""
## Obtener los datos

Los datos los puedes descargar desde el [github](https://raw.githubusercontent.com/AprendizajeAutomaticoUJI/DataSets/master/evolucion_del_euribor_mensual.csv) de la asignatura.

Es un fichero CSV donde los valores de cada fila están separados por «;». Además, el separador de decimales es «,». Para leer los datos y convertirlos en un **Dataframe** puedes utilizar:

```julia
ruta = "https://raw.githubusercontent.com/AprendizajeAutomaticoUJI/DataSets/master/evolucion_del_euribor_mensual.csv"
datos = CSV.File(Downloads.download(ruta), delim=';', decimal=',') |> DataFrame
```

Cada dato tiene tres atributos: **Año**, **Periodo** y **Euribor**. El periodo se refiere al mes del año correspondiente.
"""

# ╔═╡ f70b0dc9-3154-46c7-bf7c-3da35a7a0d8c
md"""
## Explorar los datos para conocerlos mejor

Primero, identifica si existe alguna tendencia en los datos. Para ello calcula 
un media deslizante sobre los datos:

```julia
media_deslizante(serie, tamaño_ventana) = [sum(serie[i:i+tamaño_ventana-1])/(tamaño_ventana) for i ∈ 1:length(serie)-tamaño_ventana];
```

Estudia la autocorrelación de los datos:

```julia
using StatsBase

autocor(vector)
```
"""

# ╔═╡ 7a22cab9-5089-4b82-90ee-0f38403cf419
md"""
## Preparar lo datos para que muestren los patrones

Para preparar los datos la siguiente función te puede ser de utilidad:

```julia
function procesa_datos(euribor, tamaño_lote, fraccion_entrenamiento)
	indice = Int32(length(euribor) * fraccion_entrenamiento)
    entrenamiento = euribor[1:indice]
    prueba = euribor[indice+1:end]
    escalado = fit(UnitRangeTransform, entrenamiento)
    entrenamiento_escalado = Float32.(StatsBase.transform(escalado, entrenamiento))
    prueba_escalado = Float32.(StatsBase.transform(escalado, prueba))
    Xentrenamiento = stack([entrenamiento_escalado[i : i+(tamaño_lote - 1)] for i ∈ 1:length(entrenamiento_escalado) - tamaño_lote ])
    yentrenamiento = entrenamiento_escalado[tamaño_lote + 1 : end]'
    Xprueba = stack([prueba_escalado[i : i+(tamaño_lote - 1)] for i ∈ 1:length(prueba_escalado) - tamaño_lote])
    yprueba = prueba_escalado[(tamaño_lote + 1):end]'
    return Xentrenamiento, yentrenamiento, Xprueba, yprueba, escalado
end
```

**euribor** es el Dataframe que contiene los datos del Euribor y **tamaño_lote** es el tamaño de cada una de las secuencias que quieres generar. También debes proporcionar la fracción de datos de entrenamiento. Presta atención a los valores de retorno de esta función **escalado**.
"""

# ╔═╡ 3a3196e1-32dc-42dd-ab2a-51dddbc62ef7
md"""
## Crear una primera versión del modelo

Crea primero una RNN con una única capa. Elige el número de neuronas dentro de la capa. Entrena el modelo.

!!! info "Advertencia"
	Debes resetear el modelo antes de calcular las pérdidas:
	```julia
	function perdidas(modelo, X, y)
	    Flux.reset!(modelo)
	    Flux.Losses.mse(modelo(X), y)
	end
	```

Estudia la historia del entrenamiento para ver si hay sobreentrenamiento.

Amplia el modelo para añadir más capas recurrentes.

Repite el proceso, crea una red LSTM con una única capa y procede como en el caso de la red recurrente.

Amplia el modelo para añadir más capas LSTM.
"""

# ╔═╡ f3fb9f32-4930-448f-8975-6a06c7805908
md"""
## Ajustar el modelo para obtener una solución

En ambas redes, estudia como varía en error cuadrático medio entre los valores reales y las predicciones dependiendo de parámetros tales como el número de neuronas en cada capa, y el número de capas.

¿Se te ocurre algún otro tipo de capa que puedes incluir para intentar mejorar el modelo?
"""

# ╔═╡ 2d2009a0-ff0d-4319-8dc9-f4e07c9ad8e4
md"""
## Presentar la solución

Presenta las principales conclusiones y respáldalas con los análisis que has realizado.
"""

# ╔═╡ 4f0b83f4-145b-4186-9661-3d40e34eb509
md"""
## Critica del trabajo y posibles mejoras

A la luz de todos los resultados que has obtenido, ¿cómo podrías seguir mejorando tu modelo?
"""

# ╔═╡ 2ab4b0d1-4549-4aeb-8302-b20da7a40eb6
md"""
# Entrega

El trabajo que debes entregar para su corrección es el libro de notas de Pluto (fichero con extensión jl).

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

julia_version = "1.12.2"
manifest_format = "2.0"
project_hash = "701fb2bf464f8aa820d1ec11f0c1b4d98e4b75fe"

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

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

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

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
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
# ╟─3aa149be-9d44-11f0-17aa-eb647770c269
# ╟─54cd69eb-b10b-4dbe-86d7-18493250efac
# ╟─c4c50b45-e3c6-4ed8-9d98-596f0d30f419
# ╟─a1758fe4-7afe-42ff-835c-4ff710b145eb
# ╟─c03e3d69-6460-4608-8a1e-fe64e40146ec
# ╟─40722903-fe77-41cd-8d4f-6ba87c46dec9
# ╟─565b929f-b551-47ba-b05b-10494540fbec
# ╟─f7bb6033-92c4-4dd1-8a28-e7449ea6b036
# ╟─e2f1cead-58cb-49e0-8908-80e0477a2441
# ╟─2a622058-e74a-4a03-8299-d0d71a180871
# ╟─0f9d6c20-5ba9-4da4-8225-f7e6a1d33cca
# ╟─b6ef3b13-54fc-4f81-a544-7ae49fca2d2b
# ╟─f70b0dc9-3154-46c7-bf7c-3da35a7a0d8c
# ╟─7a22cab9-5089-4b82-90ee-0f38403cf419
# ╟─3a3196e1-32dc-42dd-ab2a-51dddbc62ef7
# ╟─f3fb9f32-4930-448f-8975-6a06c7805908
# ╟─2d2009a0-ff0d-4319-8dc9-f4e07c9ad8e4
# ╟─4f0b83f4-145b-4186-9661-3d40e34eb509
# ╟─2ab4b0d1-4549-4aeb-8302-b20da7a40eb6
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
