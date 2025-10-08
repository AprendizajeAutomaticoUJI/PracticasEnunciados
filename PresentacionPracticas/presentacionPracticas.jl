### A Pluto.jl notebook ###
# v0.20.8

using Markdown
using InteractiveUtils

# ╔═╡ 0b911bfa-8724-11f0-00a2-8fd8ac4db5fc
using PlutoUI # Interactividad con Pluto

# ╔═╡ cd2961c0-65a8-486d-9253-bedaed423ea2
using LinearAlgebra # Algebra lineal en Julia

# ╔═╡ 06aeb727-e150-4f7c-8935-661b2960195a
using DataFrames # Trabajo con Dataframes

# ╔═╡ d2621297-9fa2-4c46-91e7-c51444d3b887
using HTTP # Peticiones HTTP

# ╔═╡ 9f0040cc-3e89-4d06-b170-e1e6e75aedf0
using CSV # Trabajo con ficheros CSV

# ╔═╡ 284ee5a5-72a6-476e-9292-b7fa1943cd29
using Plots # Visualiza gráficos

# ╔═╡ 4f3d4b25-349d-4976-8df9-3f2cc66b1d1a
TableOfContents(title="Contenidos", depth=1)

# ╔═╡ 28685d13-736b-49b9-a849-1d0ff17bad27
md"""
# Presentación de las prácticas

Óscar Belmonte Fernández - IR2130 Aprendizaje Automático

Grado en Inteligencia Robótica - Universitat Jaume I (UJI)
"""

# ╔═╡ 16cc8dd8-874e-4dab-bece-c19393eb692e
Resource(
	"https://belmonte.uji.es/imgs/uji.jpg",
	:alt => "Logo UJI",
	:width => 400
)

# ╔═╡ 5fb9ecbb-d12f-4ddc-b3ef-668e400d0fd5
md"""
Paquetes a utilizar
"""

# ╔═╡ 7ad32383-09f9-4aa0-826a-9d836149f98e
md"""
# Introducción

En esta práctica vas a familiarizarte con el lenguaje de programación Julia y con el entorno de desarrollo de libros de notas Pluto.
"""

# ╔═╡ 6ea6788b-2a99-4926-9a3f-76f126cc1746
md"""
# Objetivos de aprendizaje

1. Conocer las características del lenguaje de programación Julia.
1. Conocer el sistema de paquetes en Julia.
1. Crear un primer libro de notas con Pluto.
"""

# ╔═╡ ac2c34c8-5b0c-4886-a72f-7d46ae8e93e5
md"""
# El lenguaje de programación Julia
"""

# ╔═╡ d087111e-9c41-4640-b19c-0c3b8f83ac47
md"""
## Características de Julia

[Julia](https://julialang.org) es un lenguaje de programación de propósito general, muy rápido en ejecución, con una sintaxis orientada al cálculo y especialmente bien orienta al tratamiento de datos.

Las principales características de Julia son:

* Sigue el paradigma de programación funcional.
* Es un lenguaje compilado, pero la compilación se hace en el momento de ser necesaria (JIT compilation - Just in Time compilation).
* La sintaxis está orientada a las matemáticas.
* Utiliza tipos, pero gracias a la inferencia no es necesario definir tipos en la mayoría de las ocasiones.
* Es muy rápido en ejecución, compite con Fortran y C/C++.
* Utiliza multi-dispatch (análogo a la sobrecarga de métodos en Java).
"""

# ╔═╡ 9fc9762d-9745-4708-83e8-6786b8eeba4f
md"""
## El REPL de Julia

Hay varios modos de interaccionar con Julia, uno de ellos es el REPL: Read-Evaluate-Print-Loop. Para entrar en este modo, abre una terminal y escribe julia.
"""

# ╔═╡ a11c1299-ccaa-4fcc-a621-d45d435ff97c
Resource(
	"https://belmonte.uji.es/Docencia/IR2130/Practicas/PresentacionPracticas/Imagenes/julia-repl.png",
	:alt => "El REPL de Julia",
	:width => 500
)

# ╔═╡ 1a3091d9-9d4d-4a66-86a9-8056427e94f2
md"""
Cualquier fragmento de código que escribamos en el RPEL será evaluado después de pulsar la tecla *Enter*.
"""

# ╔═╡ 2ce5056a-4577-46ca-96c5-4f1e872fcdcd
md"""
## El gestor de paquetes de Julia

Podemos entrar en el modo de gestor de paquetes pulsando la combinación de teclas «Alt + ]»
"""

# ╔═╡ 7a16976a-9d7c-4b0d-850b-417fbb4ebb3e
Resource(
	"https://belmonte.uji.es/Docencia/IR2130/Practicas/PresentacionPracticas/Imagenes/gestor-paquetes-julia.png",
	:alt => "Gestor de paquetes de Julia",
	:width => 500
)

# ╔═╡ 89fe9146-356d-4113-ad76-52e745bfea71
md"""
## Los entornos virtuales en Julia
"""

# ╔═╡ 316595fc-bad5-4808-9d4a-08ede08d0d29
md"""
Es muy importante que actives un espacio virtual para gestionar los paquetes de un proyecto. Cada entorno virtual utiliza un conjunto de bibliotecas independiente de cualquier otro entorno virtual, de manera que sólo utilizamos los que necesitamos en cada uno de nuestros proyectos.

Para activar el directorio actual como entorno virtual escribimo: activate . (fíjate en el punto al final).
"""

# ╔═╡ eda55678-8d10-4d8e-88e6-946d5e8ad4e5
Resource(
	"https://belmonte.uji.es/Docencia/IR2130/Practicas/PresentacionPracticas/Imagenes/activar-entorno-virtual.png",
	:alt => "Activar entorno virtual en Julia",
	:width => 500
)

# ╔═╡ c3115f72-6b95-492f-8004-511d802c7d02
md"""
Fíjate en que el nombre del directorio actual aparece al inicio de la línea; esto significa que hemos creado un nuevo entorno virtual donde podemos indicar qué bibliotecas queremos utilizar en nuestro proyecto.
"""

# ╔═╡ 453c8f5d-8c22-40c5-9b00-82dec56ae82a
md"""
## Añadir bibliotecas al entorno virtual

Para añadir una biblioteca al entorno virtual, desde el modo de gestión de paquetes (pkg) escribimos: add Pluto:
"""

# ╔═╡ dea06487-6bad-4ed7-8bf9-196505d1e21b
Resource(
	"https://belmonte.uji.es/Docencia/IR2130/Practicas/PresentacionPracticas/Imagenes/añadir-biblioteca.png",
	:alt => "Añadir una biblioteca al entorno virtual",
	:width => 500
)

# ╔═╡ a0b6c71e-6927-4517-a69d-0c08d53dc644
md"""
En este caso hemos añadido la biblioteca Pluto a nuestro entorno virtual.

El gestor de bibliotecas descargará la biblioteca que le has indicado y todas las bibliotecas de las que depende. Esta acción puede llevar un poco de tiempo la primera vez, pero esta acción sólo se hace una vez, o cuando hay una actualización de Julia o de las propias bibliotecas.
"""

# ╔═╡ d76e22cf-a8d0-4fc7-8884-1857c04cb53e
md"""
# Libros de notas con Pluto

Pluto nos permite crear libros de notas, parecidos a los libros de notas Jupyter que quizás hayas utilizado con Python. De hecho, Jupyter está formado por Ju(lia)pyt(hon)r(R); los libros de notas de Jupyter se pueden usar con estos tres lenguajes de programación siempre que tengamos el kernel adecuado instalado.

Para empezar a utilizar libros de notas con Pluto:
1. Sal del modo gestor de paquetes pulsano la tecla de borrar.
2. En el REPL de julia escribe:
```shell
import Pluto
Pluto.run()
```
"""

# ╔═╡ 6f786205-7f8e-44bc-aa86-77002f32743a
Resource(
	"https://belmonte.uji.es/Docencia/IR2130/Practicas/PresentacionPracticas/Imagenes/ejecutar-pluto.png",
	:alt => "Ejecutar Pluto",
	:width => 600
)

# ╔═╡ 77d18a57-8a5a-4b57-ad9f-691e81bbb47a
md"""
Se abrirá tu navegador por defecto mostrándote una ventana parecida a la siguiente:
"""

# ╔═╡ a9de6bc4-d10b-48f3-9a8c-2137e02ae161
Resource(
	"https://belmonte.uji.es/Docencia/IR2130/Practicas/PresentacionPracticas/Imagenes/wellcome-to-pluto.png",
	:alt => "Página inicial de Pluto",
	:width => 700
)

# ╔═╡ 8dc7f440-9443-4916-8e02-97d844522614
md"""
Pulsa sobre _Create a **new notebook**_ para crear un nuevo libro de notas:
"""

# ╔═╡ bea34a3f-37ee-41f6-8654-092d72b487d8
Resource(
	"https://belmonte.uji.es/Docencia/IR2130/Practicas/PresentacionPracticas/Imagenes/nuevo-libro-pluto.png",
	:alt => "Nuevo libro de notas Pluto",
	:width => 700
)

# ╔═╡ 9271ea01-57f5-4acf-8958-6119ab13f122
md"""
## Características de los libros de notas Pluto

Los libros de notas Pluto son interactivos, cada vez que haces un cámbio en el código de una celda, el resto de celdas que tienen referencias a la celda cambiada se actualizan automáticamente.
"""

# ╔═╡ 6fddc634-f4e6-4a18-82ff-76e2d29c8016
a = 1

# ╔═╡ ae258a69-9d7a-4f4d-8dee-53e83e0079ad
b = 2

# ╔═╡ 986d4c4f-e2e6-4dcd-85d3-999998c31f77
resultado = a + b

# ╔═╡ 1adfba2f-15b8-43c0-8278-9d4a05f415d3
md"""
!!! info "Muy importante"
Debes tener el cuenta lo siguiente:

1. Sólo puedes tener una línea de código en cada celda.
1. Si quieres tener más líneas, las debes anidar dentro de **begin...end** o dentro de **let...end**
1. Hay celdas de código, y celdas donde puedes escribir con sintaxis Markdown. Para ingresar código con sintaxis Markdown pulsa *Ctrl + m* dentro de la celda.
1. *Shift + Enter* ejecuta la celda.
1. *Ctrl + Enter* ejecuta la celda y crea una nueva celda debajo.

Pero siempre es más interesante que utilices funciones.
"""

# ╔═╡ c5e939b9-f8cc-4952-9623-3846701b0b47
md"""
# Un vistazo muy rápido a Julia
"""

# ╔═╡ 0bcb2d76-ab33-4567-a64b-7c6601306dd9
md"""
## Declaración de variables

En Julia una variable se declara con un nombre y asignandoles un valor:
"""

# ╔═╡ a9c66330-5314-4e3b-9583-31b63806b7a0
peso = 75

# ╔═╡ f65b2469-9ac2-490f-a63c-fd895004e45f
md"""
Fíjate en que no hemos definido el tipo de esta variable, sin embargo Julia infiere el tipo de la variable.

El tipo de una variable lo podemos consultar así:
"""

# ╔═╡ ad08abae-daba-4c86-9582-95f93fe37703
typeof(peso)

# ╔═╡ 224a4ed9-668d-4650-ab09-96907324f9a6
md"""
El tipo inferido para peso es Int64.

Si definimos otra variable de este modo:
"""

# ╔═╡ f21b5b0a-abe9-4751-9bb8-9f204f113650
altura = 1.70

# ╔═╡ d47be41f-c699-4341-8529-4b49c8993299
typeof(altura)

# ╔═╡ 8a520e11-1dcf-43a7-ab07-51fcdfe0cf2d
md"""
Vemos que en este caso el tipo inferido es Float64.

Si queremos, podemos indicar el tipo de la variable en el momento de su declaración:
"""

# ╔═╡ fd196361-91d3-4f22-a57a-29a6ef11573f
peso_real::Float64 = 75

# ╔═╡ a413befd-a3b5-4548-9d91-5e2a9275cd7f
typeof(peso_real)

# ╔═╡ 3ff5372c-90d3-4d48-b573-16a95176ec49
md"""
## Promoción de tipos

Si en una expresión tenemos variables de distintos tipos, el resultado final es la promoción de los tipos a aquel de mayor precisión:
"""

# ╔═╡ 5f109080-057d-4b23-95ad-7f7e88f4c000
typeof(peso + peso_real)

# ╔═╡ 816c359c-2230-4d76-aac8-71741d3bc9d0
md"""
En esta [entrada](https://docs.julialang.org/en/v1/manual/types/) del manual de Julia puedes encontrar todo lo referente a tipos en Julia.
"""

# ╔═╡ 16e7e834-32ff-426c-9f35-94e52efdfc4b
md"""
## Vectores

Un vector fila se define del siguiente modo:
"""

# ╔═╡ 291a83af-695d-4092-b5b0-b7802e57feb5
w = [1 2 3]

# ╔═╡ ceee8188-b782-4973-acda-1f0fa7776251
md"""
Un vector columna se define como:
"""

# ╔═╡ 985df0cf-7d90-4946-b4f3-b4aa5c8ff1d0
v = [1, 2, 3]

# ╔═╡ b63bf834-0e8a-4256-881f-7c833854f006
md"""
Si multiplicamos un vector fila por un vector columna obtenemos un número:
"""

# ╔═╡ 709550c2-f1a3-435c-b12c-392ff0cb1cdf
w * v

# ╔═╡ da178ede-538c-4f81-a9b1-37ad877006a0
md"""
Si multiplicamos un vector columna por un vector fila obtenemos una matriz:
"""

# ╔═╡ cac2e919-57ce-4928-ab9b-d0c3f2505d08
v * w

# ╔═╡ 61634d4a-c260-49d3-a692-f33ee1326fa3
md"""
!!! danger "Cuidado!!!"
	Los índices en Julia empiezan en 1.
"""

# ╔═╡ 582fe7c8-add9-47a6-9e05-26d308f07099
# v[0] # Error
v[1] # Primer elemento

# ╔═╡ 7f2dae36-59c1-48c3-b6d3-e01254a759f2
v[end] # Último elemento

# ╔═╡ 087c52bf-abd3-445c-ba82-275a4b748dcf
 md"""
## Matrices

También podemos definir matrices del siguiente modo (por filas):
"""

# ╔═╡ a856dd86-3653-4b96-8b10-6972fb79fd66
m = [1 2 3;
	 4 5 6;
	 7 8 9]

# ╔═╡ 277c9da5-3adb-4927-a737-9d1763030e86
md"""
La traspuesta de una matriz se calcula con la notación matemática (mola):
"""

# ╔═╡ 76ed9bcd-35e6-44f9-ad98-6d937b370d22
m'

# ╔═╡ 1073e5b5-1847-4185-9bf2-30be94bd284b
md"""
Si queremos más funciones algebráicas tenemos que importar la biblioteca **LinearAlgebra**: (te recomiendo de hagas using/import de las bibliotecas al inicio de tu libro de notas).
"""

# ╔═╡ aec1fbf4-88a1-42a6-8be7-0bac614ab362
md"""
Para calular el determinante:
"""

# ╔═╡ e7aa4e6f-4041-47a1-98d3-b353da394b20
det(m)

# ╔═╡ 0fb54636-1b21-4b18-9581-c172d634c4f4
md"""
Para calcular la traza:
"""

# ╔═╡ ab21b257-9795-4f6a-a037-a2f38c221da8
tr(m)

# ╔═╡ f781b4f4-5516-4179-a424-3c363003e3dd
md"""
Si queremos sumar elemento a elemento de dos matrices:
"""

# ╔═╡ 8760e35d-3f2c-4d3b-9925-9eb780ce5218
m .+ m'

# ╔═╡ 0432cf26-3512-4cfe-9b5f-58c4a9e181f9
md"""
Veremos más sobre el operador de **broadcasting** (el punto en .+) más adelante.
"""

# ╔═╡ bda972e1-f11d-434f-9226-e1a017aaa321
md"""
# Funciones

Definir una función es muy sencillo:
"""

# ╔═╡ 537d2afd-17c6-4913-88ac-0236421ef43c
function suma(a, b)
	return a + b
end

# ╔═╡ d69794db-5ec8-4d3f-abbe-b84f89ab6a2d
md"""
La función suma la hemos definido sin indicar el tipo de datos de sus argumentos.

Esta función la podemos utilizar como culaquier otra función en Julia:
"""

# ╔═╡ 87972b7d-f466-40e8-8cef-4cce4d68d489
md"""
También podemos definir la misma función pero indicando el tipo de los argumentos:
"""

# ╔═╡ 5e38b906-b0d5-4f6d-b82a-4130ee8e3991
function suma(a::Int64, b::Int64)::Int64
	return a + b
end

# ╔═╡ d366e833-244b-48fc-8fea-317e3a93855f
md"""
Que, de nuevo, podemos utilizar así:
"""

# ╔═╡ 716c286c-8f55-4f88-88d7-5ff169549b65
md"""
Parece que todo ha ido igual que antes, pero no es así, ahora, el código que se ha ejecutado es el de la función donde indicamos el tipo de los argumentos porque es la función con los tipos más específicos.

De igual modo, podemos definir la función indicando que trabaja con argumentos de tipo Float64:
"""

# ╔═╡ 67ae26fe-daa3-4783-871b-05f65b0041f7
function suma(a::Float64, b::Float64)::Float64
	return a + b
end

# ╔═╡ 4ef4e1a4-3fec-4cac-8a24-1c710189b144
suma(1, 2)

# ╔═╡ 266ee322-fb50-4c54-8f7d-44e931d1a25b
suma(1, 2)

# ╔═╡ c9295c0d-9993-4031-8db3-a35ece8cec87
md"""
Que podemos utilizar así:
"""

# ╔═╡ 80729368-27db-416b-9936-728805e25e22
suma(1.0, 2.0)

# ╔═╡ 1d64ba22-fa3e-46f1-a5d9-e08c7e1857c2
md"""
Podemos consultar los **métodos** que implementan una función con la función **methods**:
"""

# ╔═╡ 4f37e0f0-b4b3-4bdf-9201-e358af8889c9
methods(suma)

# ╔═╡ e9f2fd9d-1421-4363-a0b4-f2a2b05810dc
md"""
Fíjate en que la función suma está implementada por tres métodos distintos, según el tipo de sus argumentos, y que cada uno de estos métodos se llamará dependiendo del tipo de sus argumentos.

Este mecanismo se conoce con el nombre del **multiple dispatch**, una misma función está implementada de distintos modos, y entre ellos se distinguen por el tipo de sus argumentos (si vienes del mundo Java es análogo a la sobrecarga de métodos)

Por curiosidad echa un vistazo a los métodos definidos de la funcion +.
"""

# ╔═╡ c1d30003-4150-44a1-9507-3fba0606ecda
md"""
## Programación funcional

Todo en Julia son funciones, las funciones son el ladrillo básico en programación en Julia.

Podemos tener funciones que reciben como argumento otras funciones; o funciones que devuelven funciones.
"""

# ╔═╡ e387e172-e5fd-47ca-b38a-77ba73b444a4
aplica(a, b, funcion) = funcion(a, b)

# ╔═╡ d38452b0-1244-4be6-a5e3-4dd4b2125506
md"""
Aquí estamos definiendo una función en una única línea, y la podemos usar como:
"""

# ╔═╡ e9f73d40-dcf9-4c5f-a77a-503962139b7b
aplica(1, 2, suma)

# ╔═╡ 4fd1b774-b784-4e58-a54c-7fab4276969a
resta(a, b) = a - b

# ╔═╡ 28333c0b-ac46-4278-8f94-cec6590fc2d7
md"""
Veamos ahora el caso de una función de que devuelve una función:
"""

# ╔═╡ 95b24ff4-ac4f-4fa2-af5e-026db39725d1
function devuelve_funcion(selector::String)::Function
	if selector == "suma"
		return suma
	elseif selector == "resta"
		return resta
	end
end

# ╔═╡ b52d3839-c73f-4d22-88f2-0ee29f699e26
devuelve_funcion("suma")(2, 2)

# ╔═╡ 9dce9605-621d-4236-a587-1b799c362ce0
devuelve_funcion("resta")(2, 2)

# ╔═╡ d10cb521-327a-4345-8abb-3d4deb259338
md"""
## Broadcasting

El operador de broadcasting es muy útil cuando trabajamos con vectores y queremos aplicar una función a cada uno de los elementos del vector:
"""

# ╔═╡ c6be9611-2bac-4f0a-9e3f-ea90a205484b
operacion(x) = x^2 + 2x + 3 # Fíjate en que no hemos escrito el símbolo de multiplicación entre el 2 y la «x»;)

# ╔═╡ fe8d51f9-356d-4d2b-820e-6b2059162dba
operacion([1, 2, 3]) # Obtenemos un error

# ╔═╡ c976bb64-7e55-447a-8bb9-a344018810b9
md"""
Una operación de *broadcasting* de una función sobre un conjunto iterable se indica con un punto detrás de la función, y antes del paréntesis con la lista de argumentos:
"""

# ╔═╡ 60f99f20-a160-4f3f-95b1-74a619474a82
operacion.([1, 2, 3])

# ╔═╡ 90ec341d-a232-4675-8049-b69569724791
md"""
## Comprenhension

Podemos crear vectores a partir de datos iterables con la siguient sintaxis
"""

# ╔═╡ 7d59c291-fc19-4e84-b805-35e84077d9c2
[operacion(x) for x in v]

# ╔═╡ 37f0a519-d659-41d8-a91c-e94444cc8b22
md"""
También podemos utilizar directamente una expresión:
"""

# ╔═╡ 2878a6f0-5370-417b-a23a-b0cac503e531
[2x-1 for x in v]

# ╔═╡ ffc6695c-af40-44bd-b4d3-9c9a565db53a
md"""
## Algunas utilidades

Crear una matriz identidad:
"""

# ╔═╡ 670109c7-d764-44a8-91d6-b60c72e08d47
I(3)

# ╔═╡ 0293d2e4-47ae-4afa-b5c7-03bc60d3f40b
md"""
Crear una matriz con unos:
"""

# ╔═╡ 2757f3c0-f489-4e34-88f8-1ad80a824506
ones(3, 3)

# ╔═╡ d8c37b7e-f25e-44f2-b4df-3179f2e2ddf2
md"""
Crear una matriz con ceros:
"""

# ╔═╡ 1f38c8f4-b29e-4176-aadd-21f6e24a1646
zeros(3, 3)

# ╔═╡ 61896f16-d6cb-404f-8463-40c06af0ad67
md"""
Invertir una matriz:
"""

# ╔═╡ 9f352c13-e689-4f6b-8a83-9ca24e7613e2
original = 2I(3) + ones(3, 3)

# ╔═╡ 19658b52-c52f-4773-9b87-e419c05fdda6
inversa = inv(original)

# ╔═╡ e99257ec-7f4e-4bb2-9d23-b985c221faa5
original * inversa

# ╔═╡ 2e0a084e-7212-45c5-bff6-daab1a1a2d7e
md"""
Podemos utilizar símbolos UTF-8
"""

# ╔═╡ 0292160f-8d7b-4fc1-adc0-767d0bbfc5e1
σ = 1.0

# ╔═╡ c10b9e43-7cbd-482a-8bfe-22ce4ac092e1
2σ

# ╔═╡ 494dac4f-010b-4cd0-b920-1a2c38f5a1dc
md"""
Incluso podemos utilizar emoticonos:
"""

# ╔═╡ 2b870c75-8ef9-483f-9e53-ee787928a694
🐶 = 100

# ╔═╡ aa8d9771-240e-4136-96c0-7ef11b07663f
🐱 = 200

# ╔═╡ 9b887aac-872b-4d95-867a-5f981706f1a6
animales = 🐶 + 🐱

# ╔═╡ 542cac61-e174-4c0e-b05a-a8d0cced5080
md"""
# Algunos paquetes

Existe una gran catidad de paquetes en Julia. Aquí unicamente vamos a ver algunos de los que más vamos a utilizar.

Vamos a empezar con el paquete [**DataFrames**](https://dataframes.juliadata.org/stable/) que nos permite trabajar con datos en forma de tabla.

Definimos la url desde donde queremos descargar los datos:
"""

# ╔═╡ f52daf9d-8d14-4c0c-8f7d-1c231cbb454a
path = "https://raw.githubusercontent.com/AprendizajeAutomaticoUJI/DataSets/refs/heads/master/Howell1.csv"

# ╔═╡ 09c7e59e-ca03-4c54-95f6-ad9e9f3ce1de
md"""
Hacemos la descarga y creamos un DataFrame a partir de los datos descargados:
"""

# ╔═╡ 038e1dc9-225d-44cf-8284-3ef2e0e13df2
datos = DataFrame(CSV.File(HTTP.get(path).body))

# ╔═╡ 8dc32ec2-7533-4734-b4e3-18bc1f31a374
md"""
Podemos obtener una descripción de cada una de las columnas del **DataFrame** con:
"""

# ╔═╡ f879d8b3-e3a1-4fd9-b9b2-ef494280700c
describe(datos)

# ╔═╡ 7e6dbdf8-5d03-4a27-bb54-e9892c755978
md"""
También podemos añadir nuevas columnas al **DataFrame**:
"""

# ╔═╡ 21b5e94f-a7b7-405d-bad6-0abecd0771cd
datos[!, :heightweight] = datos[:, :height] .* datos[:, :weight]

# ╔═╡ 7be7355c-2fd1-4ba3-b55d-c7fb1861708a
md"""
Ahora el **DataFrame** tiene una nueva columna:
"""

# ╔═╡ 90a69721-404a-4423-a089-e2af30071d9d
datos

# ╔═╡ cc654dcd-d3df-45b1-9536-5721a7e3c37e
md"""
Usando el paquete [Plots](https://docs.juliaplots.org/stable/) Podemos visualizar los datos en un gráfico de puntos (scatter plot):
"""

# ╔═╡ f8fd8f42-70af-48dc-922a-e0c6418c0991
scatter(datos.weight, datos.height)

# ╔═╡ 7338f4f0-1421-47f9-9a85-a1243391047c
md"""
Podemos visualizar el gráfio de una función anónima
"""

# ╔═╡ 45444b94-b513-4326-b4f0-06cebc6165be
plot(x -> 2x)

# ╔═╡ 82a7ab45-c99d-4c3a-881d-c1e390062a00
md"""
Y visualizar funciones predefinidas:
"""

# ╔═╡ d39680df-f47d-420a-8d18-34182de9dcc7
plot(sin)

# ╔═╡ 92680b16-8a85-410b-b708-ed179692d023
md"""
Igualmente sobre nuestras propias funciones:
"""

# ╔═╡ cfc9e041-95a3-4b34-84ca-54a7b540c2b9
plot(operacion)

# ╔═╡ 6fff972d-cb6a-45f6-b2b3-9c1c6bb38187
md"""
Poco a poco irás conociendo otros muchos paquetes de utilidad en Julia.
"""

# ╔═╡ a0c31424-f52e-449f-80b5-f6e3178df41b
md"""
# Resumen

Julia mola mucho 😍
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
CSV = "~0.10.15"
DataFrames = "~1.7.1"
HTTP = "~1.10.17"
Plots = "~1.40.19"
PlutoUI = "~0.7.68"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.7"
manifest_format = "2.0"
project_hash = "218a47d1458723ca42056d7e063318d8159b7bba"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "a656525c8b46aa6a1c76891552ed5381bb32ae7b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.30.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "0037835448781bb46feb39866934e243886d756a"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "a37ac0840a1196cd00317b57e39d6586bf0fd6f6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.7.1"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "6c72198e6a101cccdd4c9731d3985e904ba26037"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.1"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7bb1361afdb33c7f2b085aa49ea8fe1b0fb14e58"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.1+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "83dc665d0312b41367b7263e8a4d172eac1897f4"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.4"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "3a948313e7a41eb1db7a1e733e6335f17b4ab3c4"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "7.1.1+0"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "1828eb7275491981fa5f1752a5e126e8f26f8741"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.17"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "27299071cc29e409488ada41ec7643e0ab19091f"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.17+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "35fbd0cefb04a516104b8e183ce0df11b70a3f1a"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.84.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "ed5e9c58612c4e081aecdb6e1a479e18462e041e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

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

[[deps.InlineStrings]]
git-tree-sha1 = "8f3d257792a522b4601c24a577954b0a8cd7334d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.5"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e95866623950267c1e4878846f848d94810de475"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.2+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "52e1296ebbde0db845b356abbbe67fb82a0a116c"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.9"

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

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "706dfd3c0dd56ca090e86884db6eda70fa7dd4af"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.1+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d3c8af829abaeba27181db4acb485b18d15d89c6"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.1+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.5+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "f1a7e086c677df53e064e0fdd2c9d0b0833e3f6e"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.5.0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2ae7d4ddec2e13ad3bddf5c0796f7547cf682391"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.2+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c392fc5dd032381919e3b22dd32d6443760ce7ea"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.5.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "275a9a6d85dc86c24d03d1837a0010226a96f540"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.3+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "0c5a5b7e440c008fe31416a3ac9e0d2057c81106"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.19"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "ec9e63bd098c50e4ad28e7cb95ca7a4860603298"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.68"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

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

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "eb38d376097f47316fe089fc62cb7c6d85383a52"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.8.2+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "da7adf145cce0d44e892626e647f9dcbe9cb3e10"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.8.2+1"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "9eca9fc3fe515d619ce004c83c31ffd3f85c7ccf"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.8.2+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "e1d5e16d0f65762396f9ca4644a5f4ddab8d452b"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.8.2+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "95af145932c2ed859b63329952ce8d633719f091"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2c962245732371acd51700dbb268af311bddd719"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.6"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

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

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "6258d453843c466d84c17a58732dda5deeb8d3af"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.24.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    PrintfExt = "Printf"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"
    Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "af305cc62419f9bd61b6644d19170a4d258c7967"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.7.0"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "9caba99d38404b285db8801d5c45ef4f4f425a6d"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.1+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a5bc75478d323358a90dc36766f3c99ba7feb024"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.6+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "aff463c82a773cb86061bce8d53a0d976854923e"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.5+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "e3150c7400c41e207012b41659591f083f3ef795"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.3+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "c5bf2dad6a03dfef57ea0a170a1fe493601603f2"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.5+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "00af7ebdc563c9217ecc67776d1bbf037dbcebf4"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.44.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4bba74fa59ab0755167ad24f98800fe5d727175b"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.12.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "07b6a107d926093898e82b3b1db657ebe33134ec"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.50+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "fbf139bce07a534df0e699dbb5f5cc9346f95cc1"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.9.2+0"
"""

# ╔═╡ Cell order:
# ╟─0b911bfa-8724-11f0-00a2-8fd8ac4db5fc
# ╟─4f3d4b25-349d-4976-8df9-3f2cc66b1d1a
# ╟─28685d13-736b-49b9-a849-1d0ff17bad27
# ╟─16cc8dd8-874e-4dab-bece-c19393eb692e
# ╟─5fb9ecbb-d12f-4ddc-b3ef-668e400d0fd5
# ╠═cd2961c0-65a8-486d-9253-bedaed423ea2
# ╠═06aeb727-e150-4f7c-8935-661b2960195a
# ╠═d2621297-9fa2-4c46-91e7-c51444d3b887
# ╠═9f0040cc-3e89-4d06-b170-e1e6e75aedf0
# ╠═284ee5a5-72a6-476e-9292-b7fa1943cd29
# ╟─7ad32383-09f9-4aa0-826a-9d836149f98e
# ╟─6ea6788b-2a99-4926-9a3f-76f126cc1746
# ╟─ac2c34c8-5b0c-4886-a72f-7d46ae8e93e5
# ╟─d087111e-9c41-4640-b19c-0c3b8f83ac47
# ╟─9fc9762d-9745-4708-83e8-6786b8eeba4f
# ╟─a11c1299-ccaa-4fcc-a621-d45d435ff97c
# ╟─1a3091d9-9d4d-4a66-86a9-8056427e94f2
# ╟─2ce5056a-4577-46ca-96c5-4f1e872fcdcd
# ╟─7a16976a-9d7c-4b0d-850b-417fbb4ebb3e
# ╟─89fe9146-356d-4113-ad76-52e745bfea71
# ╟─316595fc-bad5-4808-9d4a-08ede08d0d29
# ╟─eda55678-8d10-4d8e-88e6-946d5e8ad4e5
# ╟─c3115f72-6b95-492f-8004-511d802c7d02
# ╟─453c8f5d-8c22-40c5-9b00-82dec56ae82a
# ╟─dea06487-6bad-4ed7-8bf9-196505d1e21b
# ╟─a0b6c71e-6927-4517-a69d-0c08d53dc644
# ╟─d76e22cf-a8d0-4fc7-8884-1857c04cb53e
# ╟─6f786205-7f8e-44bc-aa86-77002f32743a
# ╟─77d18a57-8a5a-4b57-ad9f-691e81bbb47a
# ╟─a9de6bc4-d10b-48f3-9a8c-2137e02ae161
# ╟─8dc7f440-9443-4916-8e02-97d844522614
# ╟─bea34a3f-37ee-41f6-8654-092d72b487d8
# ╟─9271ea01-57f5-4acf-8958-6119ab13f122
# ╠═6fddc634-f4e6-4a18-82ff-76e2d29c8016
# ╠═ae258a69-9d7a-4f4d-8dee-53e83e0079ad
# ╠═986d4c4f-e2e6-4dcd-85d3-999998c31f77
# ╟─1adfba2f-15b8-43c0-8278-9d4a05f415d3
# ╟─c5e939b9-f8cc-4952-9623-3846701b0b47
# ╟─0bcb2d76-ab33-4567-a64b-7c6601306dd9
# ╠═a9c66330-5314-4e3b-9583-31b63806b7a0
# ╟─f65b2469-9ac2-490f-a63c-fd895004e45f
# ╠═ad08abae-daba-4c86-9582-95f93fe37703
# ╟─224a4ed9-668d-4650-ab09-96907324f9a6
# ╠═f21b5b0a-abe9-4751-9bb8-9f204f113650
# ╠═d47be41f-c699-4341-8529-4b49c8993299
# ╟─8a520e11-1dcf-43a7-ab07-51fcdfe0cf2d
# ╠═fd196361-91d3-4f22-a57a-29a6ef11573f
# ╠═a413befd-a3b5-4548-9d91-5e2a9275cd7f
# ╟─3ff5372c-90d3-4d48-b573-16a95176ec49
# ╠═5f109080-057d-4b23-95ad-7f7e88f4c000
# ╟─816c359c-2230-4d76-aac8-71741d3bc9d0
# ╟─16e7e834-32ff-426c-9f35-94e52efdfc4b
# ╠═291a83af-695d-4092-b5b0-b7802e57feb5
# ╟─ceee8188-b782-4973-acda-1f0fa7776251
# ╠═985df0cf-7d90-4946-b4f3-b4aa5c8ff1d0
# ╟─b63bf834-0e8a-4256-881f-7c833854f006
# ╠═709550c2-f1a3-435c-b12c-392ff0cb1cdf
# ╟─da178ede-538c-4f81-a9b1-37ad877006a0
# ╠═cac2e919-57ce-4928-ab9b-d0c3f2505d08
# ╟─61634d4a-c260-49d3-a692-f33ee1326fa3
# ╠═582fe7c8-add9-47a6-9e05-26d308f07099
# ╠═7f2dae36-59c1-48c3-b6d3-e01254a759f2
# ╟─087c52bf-abd3-445c-ba82-275a4b748dcf
# ╠═a856dd86-3653-4b96-8b10-6972fb79fd66
# ╟─277c9da5-3adb-4927-a737-9d1763030e86
# ╠═76ed9bcd-35e6-44f9-ad98-6d937b370d22
# ╟─1073e5b5-1847-4185-9bf2-30be94bd284b
# ╟─aec1fbf4-88a1-42a6-8be7-0bac614ab362
# ╠═e7aa4e6f-4041-47a1-98d3-b353da394b20
# ╟─0fb54636-1b21-4b18-9581-c172d634c4f4
# ╠═ab21b257-9795-4f6a-a037-a2f38c221da8
# ╟─f781b4f4-5516-4179-a424-3c363003e3dd
# ╠═8760e35d-3f2c-4d3b-9925-9eb780ce5218
# ╟─0432cf26-3512-4cfe-9b5f-58c4a9e181f9
# ╟─bda972e1-f11d-434f-9226-e1a017aaa321
# ╠═537d2afd-17c6-4913-88ac-0236421ef43c
# ╟─d69794db-5ec8-4d3f-abbe-b84f89ab6a2d
# ╠═4ef4e1a4-3fec-4cac-8a24-1c710189b144
# ╟─87972b7d-f466-40e8-8cef-4cce4d68d489
# ╠═5e38b906-b0d5-4f6d-b82a-4130ee8e3991
# ╟─d366e833-244b-48fc-8fea-317e3a93855f
# ╠═266ee322-fb50-4c54-8f7d-44e931d1a25b
# ╟─716c286c-8f55-4f88-88d7-5ff169549b65
# ╠═67ae26fe-daa3-4783-871b-05f65b0041f7
# ╟─c9295c0d-9993-4031-8db3-a35ece8cec87
# ╠═80729368-27db-416b-9936-728805e25e22
# ╟─1d64ba22-fa3e-46f1-a5d9-e08c7e1857c2
# ╟─4f37e0f0-b4b3-4bdf-9201-e358af8889c9
# ╟─e9f2fd9d-1421-4363-a0b4-f2a2b05810dc
# ╟─c1d30003-4150-44a1-9507-3fba0606ecda
# ╠═e387e172-e5fd-47ca-b38a-77ba73b444a4
# ╟─d38452b0-1244-4be6-a5e3-4dd4b2125506
# ╠═e9f73d40-dcf9-4c5f-a77a-503962139b7b
# ╠═4fd1b774-b784-4e58-a54c-7fab4276969a
# ╟─28333c0b-ac46-4278-8f94-cec6590fc2d7
# ╠═95b24ff4-ac4f-4fa2-af5e-026db39725d1
# ╠═b52d3839-c73f-4d22-88f2-0ee29f699e26
# ╠═9dce9605-621d-4236-a587-1b799c362ce0
# ╟─d10cb521-327a-4345-8abb-3d4deb259338
# ╠═c6be9611-2bac-4f0a-9e3f-ea90a205484b
# ╠═fe8d51f9-356d-4d2b-820e-6b2059162dba
# ╟─c976bb64-7e55-447a-8bb9-a344018810b9
# ╠═60f99f20-a160-4f3f-95b1-74a619474a82
# ╟─90ec341d-a232-4675-8049-b69569724791
# ╠═7d59c291-fc19-4e84-b805-35e84077d9c2
# ╟─37f0a519-d659-41d8-a91c-e94444cc8b22
# ╠═2878a6f0-5370-417b-a23a-b0cac503e531
# ╟─ffc6695c-af40-44bd-b4d3-9c9a565db53a
# ╠═670109c7-d764-44a8-91d6-b60c72e08d47
# ╟─0293d2e4-47ae-4afa-b5c7-03bc60d3f40b
# ╠═2757f3c0-f489-4e34-88f8-1ad80a824506
# ╟─d8c37b7e-f25e-44f2-b4df-3179f2e2ddf2
# ╠═1f38c8f4-b29e-4176-aadd-21f6e24a1646
# ╟─61896f16-d6cb-404f-8463-40c06af0ad67
# ╠═9f352c13-e689-4f6b-8a83-9ca24e7613e2
# ╠═19658b52-c52f-4773-9b87-e419c05fdda6
# ╠═e99257ec-7f4e-4bb2-9d23-b985c221faa5
# ╟─2e0a084e-7212-45c5-bff6-daab1a1a2d7e
# ╠═0292160f-8d7b-4fc1-adc0-767d0bbfc5e1
# ╠═c10b9e43-7cbd-482a-8bfe-22ce4ac092e1
# ╟─494dac4f-010b-4cd0-b920-1a2c38f5a1dc
# ╠═2b870c75-8ef9-483f-9e53-ee787928a694
# ╠═aa8d9771-240e-4136-96c0-7ef11b07663f
# ╠═9b887aac-872b-4d95-867a-5f981706f1a6
# ╟─542cac61-e174-4c0e-b05a-a8d0cced5080
# ╠═f52daf9d-8d14-4c0c-8f7d-1c231cbb454a
# ╟─09c7e59e-ca03-4c54-95f6-ad9e9f3ce1de
# ╠═038e1dc9-225d-44cf-8284-3ef2e0e13df2
# ╟─8dc32ec2-7533-4734-b4e3-18bc1f31a374
# ╠═f879d8b3-e3a1-4fd9-b9b2-ef494280700c
# ╟─7e6dbdf8-5d03-4a27-bb54-e9892c755978
# ╠═21b5e94f-a7b7-405d-bad6-0abecd0771cd
# ╟─7be7355c-2fd1-4ba3-b55d-c7fb1861708a
# ╠═90a69721-404a-4423-a089-e2af30071d9d
# ╟─cc654dcd-d3df-45b1-9536-5721a7e3c37e
# ╟─f8fd8f42-70af-48dc-922a-e0c6418c0991
# ╟─7338f4f0-1421-47f9-9a85-a1243391047c
# ╠═45444b94-b513-4326-b4f0-06cebc6165be
# ╟─82a7ab45-c99d-4c3a-881d-c1e390062a00
# ╠═d39680df-f47d-420a-8d18-34182de9dcc7
# ╟─92680b16-8a85-410b-b708-ed179692d023
# ╠═cfc9e041-95a3-4b34-84ca-54a7b540c2b9
# ╟─6fff972d-cb6a-45f6-b2b3-9c1c6bb38187
# ╟─a0c31424-f52e-449f-80b5-f6e3178df41b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
