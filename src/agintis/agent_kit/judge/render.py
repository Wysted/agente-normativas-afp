"""Renders a record for a judge: the data goes in a USER message, markup neutralized.

Question, answer and chunk texts are untrusted (the answer is model output, the chunks are
retrieved documents): `<` / `>` never survive, so nothing inside can close or fake a block. The
sources are rendered by the SAME function the synthesis used (`[n] title (locator)` + text), so a
judge sees what the generator saw.
"""

from collections.abc import Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from agintis.agent_kit.judge.citations import CitedSentence
from agintis.agent_kit.judge.records import AnswerRecord
from agintis.agent_kit.source_text import neutralize

NO_SOURCES = "(ninguna fuente)"


def render_sources(record: AnswerRecord) -> str:
    return record.rendered_sources if record.sources else NO_SOURCES


def faithfulness_messages(prompt: str, record: AnswerRecord) -> list[SystemMessage | HumanMessage]:
    return [
        SystemMessage(prompt),
        HumanMessage(
            f"<fuentes>\n{render_sources(record)}\n</fuentes>\n\n"
            f"<respuesta>\n{neutralize(record.answer)}\n</respuesta>\n\n"
            f"<pregunta>\n{neutralize(record.question)}\n</pregunta>"
        ),
    ]


def citation_messages(
    prompt: str, record: AnswerRecord, sentences: Sequence[CitedSentence]
) -> list[SystemMessage | HumanMessage]:
    by_number = {s.n: s for s in record.sources}
    blocks = []
    for sentence in sentences:
        # unreachable: the dangling precheck uses the same parse as cited_sentences
        assert all(n in by_number for n in sentence.numbers)
        cited = "\n\n".join(by_number[n].render() for n in sentence.numbers)
        blocks.append(
            f'<oracion index="{sentence.index}">\n{neutralize(sentence.text)}\n\n'
            f"Fuentes citadas:\n{cited}\n</oracion>"
        )
    return [SystemMessage(prompt), HumanMessage("\n\n".join(blocks))]


def relevance_messages(prompt: str, record: AnswerRecord) -> list[SystemMessage | HumanMessage]:
    return [
        SystemMessage(prompt),
        HumanMessage(
            f"<pregunta>\n{neutralize(record.question)}\n</pregunta>\n\n"
            f"<respuesta>\n{neutralize(record.answer)}\n</respuesta>"
        ),
    ]
