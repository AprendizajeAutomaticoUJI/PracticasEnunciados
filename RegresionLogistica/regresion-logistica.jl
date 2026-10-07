### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 9e48c088-8ccb-11f0-13de-59949736e3cf
using PlutoUI # Interactividad con Pluto

# ╔═╡ 1d352265-df23-4c93-af53-ebdc0e937f75
TableOfContents(title="Contenidos", depth=1)

# ╔═╡ e4948e5c-2afd-4b7c-9b38-853aa17c2356
md"""
# Regresión logística

Óscar Belmonte Fernández - IR2130 Aprendizaje Automático

Grado en Inteligencia Robótica - Universitat Jaume I (UJI)
"""

# ╔═╡ 78d8f43e-e2d0-43a9-b3f3-c8fcf21c4e93
Resource(
	"https://belmonte.uji.es/imgs/uji.jpg",
	:alt => "Logo UJI",
	:width => 400,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 2b49f649-dbe0-4b47-a7cf-18901e406159
md"""
# Introducción

En esta práctica vas a utilizar el algoritmo de Regresión Logística para clasificar tumores como benignos o malignos.

El modelo deberá predecir si un tumor es benigno o maligno a partir de ciertas características.

Los datos los puedes descargar desde el [github](https://raw.githubusercontent.com/AprendizajeAutomaticoUJI/DataSets/master/biopsy.csv) de la asignatura.

Vas a crear un libro de notas de Pluto para desarrollar la práctica. Este libro lo debes subir a aulavirtual para su evaluación.
"""

# ╔═╡ 36ef9e09-d0d6-4e7c-bbe0-9bb20eda5a20
md"""
!!! danger "Declaración de uso de la IA"
	Si haces uso de la IA, debes indicar claramente qué fragmentos de código has creado con su ayuda, y también debe incorparar los prompts que has utilizado. Si no incluyes decaración de uso de la IA y se detecta su uso, la prácticas será calificada como **No Apta**.
"""

# ╔═╡ b33523a2-f4e9-4e6a-a6b4-a7498658da04
md"""
# Duración de la práctica

A esta práctica le vamos a dedicar una sesión de prácticas.
"""

# ╔═╡ 089707e4-0a2b-4d69-a121-eedf27d20e60
md"""
# Objetivos de aprendizaje

* Emplear el algoritmo de Regresión Logística para realizar tareas de clasificación.
* Estimar el rendimiento de la Regresión Logística para el conjunto de datos proporcionado.
* Decidir cuál es el mejor conjunto de características para construir el clasificador.
* Argumentar las decisiones realizadas.
"""

# ╔═╡ f2595321-d4bc-4303-9d7d-afaec016e845
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

# ╔═╡ 31b2c9a5-183c-43bc-955d-0845730b0bd0
md"""
# Objetivo

Utilizar la regresión logística para clasificar un tumor como **benigno** o **maligno** a partir de un conjunto de características.
"""

# ╔═╡ a82a2083-b49f-4eaf-a2d7-0bcef7e4b8fe
md"""
# Tareas a realizar

Vas a seguir una adaptación del esquema que se presentó en el tema de **Proyectos de Aprendizaje Automático**.
"""

# ╔═╡ 181df0df-f5e5-4e8b-8cbf-4a6cb45bcab2
md"""
## Definir el problema y tener un imagen del conjunto

Describe, con tus propias palabras cuál es el problema que se pretende resolver y cuál es su alcance.
"""

# ╔═╡ 811e429b-1227-41df-a6ed-dd28710aaad4
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

# ╔═╡ f848c022-e460-40a5-ab8e-aeccb3c35437
md"""
## Explorar los datos para conocerlos mejor

Realiza un análisis exploratorio de los datos, para ello, visualiza tus datos, haz un análisis estadístico de ellos y extrae conclusiones.

Presta atención a la característica V6. A los datos faltantes se lea ha asignado el valor "NA", por eso esta columna es de tipo String.

También puedes comprobar que el conjunto de datos está desbalaceado, hay más datos de la clase «benigno» que de la clase «maligno».
"""

# ╔═╡ d1689d4a-28d5-47e6-b21e-892793d4780c
md"""
## Preparar los datos para que muestren los patrones

A parir de los resultados del apartado anterior, prepara los datos para resaltar los posibles patrones en ellos. En particular, elimina los datos con V6 igual a "NA", y cambia el tipo de los datos de esta columna, para ello puedes utilizar:

```.julia
datos[!, :V6] = parse.(Int64, datos[!, :V6])
```
"""

# ╔═╡ aa01f04e-9d46-4802-98dd-915383da8a00
md"""
Por otro lado, el modelo [LogisticClassifier](https://juliaai.github.io/MLJ.jl/dev/models/LogisticClassifier_MLJLinearModels/) del paquete [MLJ](https://juliaml.ai/), indica que en los datos de entrenamiento espera que *X* sea un DataFrame con columnas de valores Continuous, y que *y* sea un vector de OrderedFactor o Multiclass. En nuestro caso, como no hay una relación de orden entre **benigno** y **maligno** debemos elegir Multiclass.

Puedes consultar los tipos de tu data frame y la conversión que entre tipos de Julia y tipos de MLJ (scitype) se va a aplicacar con la función:

```.julia
schema(df)
```

Para cambiar los tipos del DataFrame a los tipos que MLJ espera, utiliza:

```.julia
df_coerce = coerce(df,
				  :Vx => Continuous.
				  :clase => Multiclass,
				 )
```


Además, a partir del conjunto con coerciones, debes crear dos conjuntos, una para entrenar el modelo y otro para hacer pruebas, como el conjunto de datos está desbalanceado, debes estratificar la partición de los datos, es decir, tomar del cojunto original la misma proporción de datos de las dos clases (en este caso tenemos sólo dos).

Esto lo puedes conseguir del siguiente modo:

```.julia
using MLJ

partition(df_coerce, 0.8, rng = 69, stratify = df_coerce.clase, shuffle = true)
```

donde df\_coerce es el nombre de tu DataFrame y df\_coerce.clase es el vector que contiene información de las clases del conjunto de datos
"""

# ╔═╡ 97aa8b88-436e-4932-8ae3-142c2545adcd
md"""
## Crear una primera versión del modelo

Con la información que has conseguido del análisis realizado, crea un primera versión de un regresor logístico donde la variable predictora sea sólamente **V1**, y estima cuál es su precisión (accuracy), y muestra la matriz de confusión. Ten en cuenta que con la regresión logística da un valor de probabilidad de pertenencia a cada una de las clases, puedes obtener el valor de probabilidad de la clase con mayor probabilidad con **predict_mode(maquina, X)**, como en el siguiente ejemplo:

```.julia
predicciones = predict_mode(maquina, Xprueba)
confusion_matrix(predicciones, yprueba)
```
"""

# ╔═╡ 6a6b9d7a-c9a2-48ec-9644-acd31d3c6697
md"""
El método **predict(maquina, X)** te devuelve una pareja de valores con la probabilidad de pertenencia a cada clase:

```.julia
predicciones = predict(maquina, Xprueba)
roc_curve(predicciones, yprueba)
auc(predicciones, yprueba)
```

Aunque el conjunto de datos tienen suficientes muestras, es interesante realizar una validación cruzada, generando varios experimentos y promediando los resultados. Esto se puede realizar de una manera muy sencialla del siguiente modo:

```.julia
evaluate!(
	maquina,
	resampling = StratifiedCV(nfolds = 5, rng = 69),
	measures = [log_loss, accuracy],
)
```

donde **nfolds** es el número de experimentos que vas a realizar
"""

# ╔═╡ fce036e3-cbac-4e17-bf99-077e4532dafa
md"""
## Ajustar el modelo para obtener una solución

Ahora que ya tienes una primera versión del modelo funcionando, añade nuevas características **Vx** al modelo. ¿Cuál es la primera que eliges ahora que tu modelo ya tiene **V1**? ¿Por qué eliges esa nueva característica? ¿Cuánto se ha incrementado la precisión de tu modelo? Razona el posible cambio.

Sigue añadiendo características, una tras otra, hasta que tu modelo final las incluya todas.

Haz una análisis detallado de todas las conclusiones que has extraído.
"""

# ╔═╡ 1d8c85dc-a009-441f-9bdc-c87b713bd208
md"""
## Presentar la solución

Presenta los principales conclusiones y respalda cada conclusión con los análisis que has realizado.
"""

# ╔═╡ b9a95e32-b469-424e-a501-9a6ad8634da2
md"""
## Critica del trabajo y posibles mejoras

A la luz de todos los resultados que has obtenido, ¿cómo podrías seguir mejorando tu modelo?
"""

# ╔═╡ cc81a964-0f7a-41aa-84d2-79e4e0dc2347
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

julia_version = "1.12.7"
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
version = "1.3.1+2"

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
version = "2025.11.4"

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
version = "3.5.6+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"

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
# ╟─9e48c088-8ccb-11f0-13de-59949736e3cf
# ╟─1d352265-df23-4c93-af53-ebdc0e937f75
# ╟─e4948e5c-2afd-4b7c-9b38-853aa17c2356
# ╟─78d8f43e-e2d0-43a9-b3f3-c8fcf21c4e93
# ╟─2b49f649-dbe0-4b47-a7cf-18901e406159
# ╟─36ef9e09-d0d6-4e7c-bbe0-9bb20eda5a20
# ╟─b33523a2-f4e9-4e6a-a6b4-a7498658da04
# ╟─089707e4-0a2b-4d69-a121-eedf27d20e60
# ╟─f2595321-d4bc-4303-9d7d-afaec016e845
# ╟─31b2c9a5-183c-43bc-955d-0845730b0bd0
# ╟─a82a2083-b49f-4eaf-a2d7-0bcef7e4b8fe
# ╟─181df0df-f5e5-4e8b-8cbf-4a6cb45bcab2
# ╟─811e429b-1227-41df-a6ed-dd28710aaad4
# ╟─f848c022-e460-40a5-ab8e-aeccb3c35437
# ╟─d1689d4a-28d5-47e6-b21e-892793d4780c
# ╟─aa01f04e-9d46-4802-98dd-915383da8a00
# ╟─97aa8b88-436e-4932-8ae3-142c2545adcd
# ╟─6a6b9d7a-c9a2-48ec-9644-acd31d3c6697
# ╟─fce036e3-cbac-4e17-bf99-077e4532dafa
# ╟─1d8c85dc-a009-441f-9bdc-c87b713bd208
# ╟─b9a95e32-b469-424e-a501-9a6ad8634da2
# ╟─cc81a964-0f7a-41aa-84d2-79e4e0dc2347
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
