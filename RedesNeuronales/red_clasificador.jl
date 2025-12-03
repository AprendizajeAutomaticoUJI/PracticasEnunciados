### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ b26a0738-b89e-11f0-2b16-a98168892685
using PlutoUI

# ╔═╡ 1aec551f-cbc5-4dc7-8bd4-8a29d56af21f
TableOfContents(title = "Contenidos", depth = 1)

# ╔═╡ 760e80d0-e51a-4e76-83c5-da136380ab32
md"""
# Red neuronal para clasificación

Óscar Belmonte Fernández - IR2130 Aprendizaje Automático

Grado en Inteligencia Robótica - Universitat Jaume I (UJI)
"""

# ╔═╡ 80e917d5-fdb5-4e0f-aa76-e5240d258818
Resource(
	"https://belmonte.uji.es/imgs/uji.jpg",
	:alt => "Logo UJI",
	:width => 400,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 45282827-4532-4a29-9048-c28c9a806e03
md"""
# Introducción

En esta práctica vas a crear y entrenar una rede neuronal para realizar tareas de clasificación.

Vas autilizar el mismo conjunto de datos que en las prácticas de regresión logística y SVM, lo que te permitirá comparar los resultados obtenidos entre los modelos. El modelo deberá predecir si un tumor es benigno o maligno a partir de ciertas características.

Los datos los puedes descargar desde el [github](https://raw.githubusercontent.com/AprendizajeAutomaticoUJI/DataSets/master/biopsy.csv) de la asignatura.

Vas a crear un libro de notas de Pluto para desarrollar la práctica. Este libro lo debes subir a aulavirtual para su evaluación.
"""

# ╔═╡ 60f1614b-9995-4604-abeb-390a473df3d7
md"""
# Duración de la práctica

A esta práctica le vamos a dedicar una sesión.
"""

# ╔═╡ f61ff3a9-3666-4ab9-8c1d-55606a6e72bd
md"""
# Objetivos de aprendizaje

* Crear una red neuronal seleccionando el número de capas, de neuronas en cada capa, las funciones de activación y los optimizadores.
* Evaluar el rendimiento de la red para distintas combinaciones de los parámetros anteriores.
* Comparar los resultados con los resultados de la práctica de regresión losgística y Máquinas de Soporte Vectorial (SVM).
"""

# ╔═╡ 509cf98c-9507-4a86-b924-3f6d75825349
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

# ╔═╡ b5cf7dda-6b01-428a-b37c-533e1e5c41b5
md"""
# Objetivo

Utilizar una red neuronal para clasificar un tumor como **benigno** o **maligno** a partir de un conjunto de características.
"""

# ╔═╡ 49e47a6d-61a3-4a85-afb1-b6182b608fbe
md"""
# Tareas a realizar

Vas a seguir una adaptación del esquema que se presentó en el tema de **Proyectos de Aprendizaje Automático**.
"""

# ╔═╡ be55aa9f-3949-49aa-9f4f-8bed90450613
md"""
## Definir el problema y tener un imagen del conjunto

Describe, con tus propias palabras cuál es el problema que se pretende resolver y cuál es su alcance.
"""

# ╔═╡ 8c33d736-ba62-4b7a-9b33-8f55ec457083
md"""
## Obtener los datos

Los datos los puedes descargar desde el [github](https://raw.githubusercontent.com/AprendizajeAutomaticoUJI/DataSets/master/biopsy.csv) de la asignatura.

Esta es la descripción de las cada una de las características de los datos:

- **ID**: Código de identificación (puede que no sea único). 
- **V1**: Grosor de los lóbulos. 
- **V2**: Uniformidad del tamaño celular. 
- **V3**: Uniformidad de la forma celular. 
- **V4**: Adhesión marginal. 
- **V5**: Tamaño de la célula epitelial única. 
- **V6**: Núcleos desnudos (faltan 16 valores). 
- **V7**: Cromatina lisa. 
- **V8**: Nucleolos normales. 
- **V9**: Mitosis. 
- **clase**: "benigno" o "maligno".  

Revisa y limpia los datos si es necesario.
"""

# ╔═╡ 2a3798cc-8249-40b7-81f3-e144fbe0f259
md"""
## Explorar los datos para conocerlos mejor

Recuerda que con **describe(df)** puedes obtener una descripción inicial de los datos.

Ya has trabajado con estos datos. A modo de recordatorio, presta atención a la característica V6. A los datos faltantes se lea ha asignado el valor "NA", por eso esta columna es de tipo String.

Además, todos los datos están en el mismo rango (1, 10), inicialmente no hace falta normalizar los datos para trabajar con redes neuronales.
"""

# ╔═╡ 540867cd-2e2c-4e06-affe-ec3c866e97ce
md"""
## Preparar lo datos para que muestren los patrones

A partir de los resultados del apartado anterior, prepara los datos para resaltar los posibles patrones en ellos. En particular, elimina los datos con :V6 igual a "NA", y cambia el tipo de los datos de esta columna, para ello puedes utilizar:

```.julia
datos[!, :V6] = parse.(Int64, datos[!, :V6])
```

Utiliza el paquete [**MLJ**](https://juliaml.ai/) para dividir el conjunto inicial de datos en dos subconjuntos, el conjunto de datos de entrenamiento (80% de los datos), y el conjunto de datos de prueba (20% restante de los datos). Recuerda especificar el origen de generación de los datos aleatorios para que tus experimentos sean reproducibles.

```julia
using MLJ: partition

entrenamiento, prueba = partition(DataFrame, 0.8, rng = 69)
```

Para entrenar la red neuronal necesitas que cada muestra de entrenamiento sea la columna de una matriz (Matrix), y que la variable a estimar sea un vector fila. Además el tipo de datos debe ser Float32. Puedes conseguir esto con:

```julia
X = Float32.(Matrix(DataFrame))' # Atención al símbolo '
```

Típicamente, cuando se utilizan redes neuronales para tareas de clasificación, lo capa de salida tiene tantas neuronas como posibles clases, por lo que tenemos que adaptar nuestras variables de salida para que sigan este criterio, para ello utilizados la *codificación one hot*, que en nuestro caso es:

```julia
y = Int32.(Flux.onehotbatch(DataFrame.clase, ("benigno", "maligno")))
```
"""

# ╔═╡ 7d840bdb-e838-4b0b-869b-c37706600107
md"""
## Crear una primera versión del modelo

Vas a utilizar el paquete Julia [**Flux**](https://fluxml.ai/) para crear y entrenar una red neuronal.

Crea una red neuronal sencilla para comprobar que la puedes entrenar sin problemas. Prueba con una única capa de 5 neuronas y función de activación *sigmoid*, y como optimizador usa **Descent**. En el caso de clasificación, la última capa debe ser una capa softmax:

Recuerda que en Flux una red se define con:

```julia
modelo = Chain(
	# Aquí cada una de las capas
	softmax
)
```

La función de pérdidas que debes utilizar en el caso de clasificación es *log cross entropy**:

```julia
perdidas(modelo, X, y) = Flux.Losses.logitbinarycrossentropy(modelo(X), y)
```

Para entrenar el modelo, el bucle básico es:

```julia
with_logger(NullLogger()) do
	for _ in 1:1000
		Flux.train!(perdidas, modelo, [(X, y)], optimizador)
	end
end
```

Utiliza el conjunto de datos de prueba para evaluar el rendimiento del modelo. Para ello tendras que desacer la coficación *one hot* y comparar con las etiquetas reales de cada muestra, esto lo puedes hacer con:

```julia
predicciones = Flux.onecold(modelo(X), ("benigno", "maligno"))
```

Donde X debe ser tu conjunto de prueba.
"""

# ╔═╡ 4b8e3d7f-e76d-46f2-8df6-be75b1349b8d
md"""
## Ajustar el modelo para mejorar la solución

Una vez que hayas comprobado que puedes entrenar el modelo y evaluar su rendimiento mostrando los resultados como una matriz de confusión, intenta mejorar el rendimiento de la red, para ello:

1. Añade más capas a la red.
1. Varía el número de neuronas en las capas.
1. Varía la función de activación.
1. Varía el optimizador.
1. Cualquier otro detalle que se te ocurra.
"""

# ╔═╡ ed5abb40-7911-4f8f-932e-95d5cd80abfa
md"""
## Presentar la solución

Haz un pequeño estudio comentando las conclusiones a las que has llegado con todos los experimentos que has hecho.
"""

# ╔═╡ 910422e6-7e79-4264-ade8-cdafc5accef3
md"""
## Críticas al trabajo y posibles mejoras

Compara los reesultados que has obtenido con el resultado que obtuviste en las prácticas de regresión logística y SVM.

¿Qué crees que se puede mejorar del trabajo que has hecho?

Haz un listado de posibles mejoras, y ordénalas por importancia de mayor a menor.
"""

# ╔═╡ 24be8b13-9d05-4329-8025-9fb84d41cdf4
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
PlutoUI = "~0.7.71"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.2"
manifest_format = "2.0"
project_hash = "4fe5cd5f5eb747cbfc9371873629578b2170de90"

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
git-tree-sha1 = "8329a3a4f75e178c11c1ce2342778bcbbbfa7e3c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.71"

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
# ╟─b26a0738-b89e-11f0-2b16-a98168892685
# ╟─1aec551f-cbc5-4dc7-8bd4-8a29d56af21f
# ╟─760e80d0-e51a-4e76-83c5-da136380ab32
# ╟─80e917d5-fdb5-4e0f-aa76-e5240d258818
# ╟─45282827-4532-4a29-9048-c28c9a806e03
# ╟─60f1614b-9995-4604-abeb-390a473df3d7
# ╟─f61ff3a9-3666-4ab9-8c1d-55606a6e72bd
# ╟─509cf98c-9507-4a86-b924-3f6d75825349
# ╟─b5cf7dda-6b01-428a-b37c-533e1e5c41b5
# ╟─49e47a6d-61a3-4a85-afb1-b6182b608fbe
# ╟─be55aa9f-3949-49aa-9f4f-8bed90450613
# ╟─8c33d736-ba62-4b7a-9b33-8f55ec457083
# ╟─2a3798cc-8249-40b7-81f3-e144fbe0f259
# ╟─540867cd-2e2c-4e06-affe-ec3c866e97ce
# ╟─7d840bdb-e838-4b0b-869b-c37706600107
# ╟─4b8e3d7f-e76d-46f2-8df6-be75b1349b8d
# ╟─ed5abb40-7911-4f8f-932e-95d5cd80abfa
# ╟─910422e6-7e79-4264-ade8-cdafc5accef3
# ╟─24be8b13-9d05-4329-8025-9fb84d41cdf4
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
