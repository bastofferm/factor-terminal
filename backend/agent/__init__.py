"""The model agent: a LangGraph workflow that chooses a security's factor model.

It replaces the hand-set estimation controls with a search that estimates,
checks and validates candidate specifications and ends in a verdict. An LLM
proposes which candidates to try next; everything that is a number - every fit,
every loss, every test, the choice of winner and the verdict - is computed in
code, on data split so the verdict is scored on a span the search never saw.
"""
