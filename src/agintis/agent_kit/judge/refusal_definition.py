"""The judges' definition of a refusal (also the only judge file the blind labeler may read)."""

REFUSAL_DEFINITION = """\
`is_refusal` es true si la RESPUESTA no entrega NINGUNA información sobre el tema preguntado: \
dice que no sabe o que no está en las fuentes, aunque agregue datos de otro tema presentados \
como "no es lo que preguntaste". NO es un rechazo: una respuesta que entrega parte de lo \
preguntado, o que indica dónde está (una entrada de índice o el número de un documento del \
mismo tema), aunque diga qué falta; tampoco una respuesta de cortesía ("De nada.", un saludo, un \
agradecimiento) ni contenido ajeno al dominio (código, consejos de salud, etc.).
Caso límite: mismo objeto, otro aspecto -> respuesta parcial, NO rechazo; un objeto vecino -> \
rechazo. Inventado: se pregunta cuánto cuesta renovar un carné de biblioteca; "se renueva en el \
mesón con la cédula [1], no se indica el costo" es parcial (viene de un documento sobre el \
tema preguntado); "no lo sé; los préstamos duran 14 días [2]" es \
rechazo (el documento trata otro objeto).
Ejemplos: "No encuentro esa información en los documentos." -> is_refusal true. \
"De nada, ¡para eso estoy!" -> is_refusal false."""
