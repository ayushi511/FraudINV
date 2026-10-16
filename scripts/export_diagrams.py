"""Write black-book diagrams as Mermaid: the agent state graph (generated from the
compiled LangGraph itself, so the diagram cannot drift from the code) and the ER diagram."""
from pathlib import Path

from agents.pipeline import FraudSystem
from backend.db import Base

out = Path(__file__).resolve().parent.parent / "docs"
out.mkdir(exist_ok=True)
S = FraudSystem.from_artifacts()
(out / "agent_graph.mmd").write_text(S.app.get_graph().draw_mermaid())

lines = ["erDiagram"]
for t in Base.metadata.sorted_tables:
    lines.append(f"    {t.name} {{")
    for c in t.columns:
        key = " PK" if c.primary_key else " FK" if c.foreign_keys else ""
        lines.append(f"        {c.type.__class__.__name__.lower()} {c.name}{key}")
    lines.append("    }")
for t in Base.metadata.sorted_tables:
    for c in t.columns:
        for fk in c.foreign_keys:
            lines.append(f'    {fk.column.table.name} ||--o{{ {t.name} : "{c.name}"')
(out / "er_diagram.mmd").write_text("\n".join(lines))
print("wrote docs/agent_graph.mmd and docs/er_diagram.mmd")
