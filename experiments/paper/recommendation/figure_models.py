"""Concrete plotting observations, panel specifications and editable layouts."""

from dataclasses import dataclass, field

from ehk.domain import CoverageRow, parse_dataclass


@dataclass(kw_only=True)
class PairedEffectRow:
    task_id: str
    stage: str
    setting: str
    sensitivity: str
    influence: float
    rewiring: float
    repost: float
    replicate: int
    epsilon: float
    history: int
    bridge_shown: int
    cross_confident: int
    updates: int
    delta_separation: float
    delta_excess: float

@dataclass(kw_only=True)
class PolicyPanel:
    title: str
    settings: tuple[tuple[float, float], ...]
    rows: tuple[PairedEffectRow, ...]
    kind: str = "policy"


@dataclass(kw_only=True)
class BridgePanel:
    title: str
    metric: str
    rows: tuple[PairedEffectRow, ...]
    kind: str = "bridge"


@dataclass(kw_only=True)
class CaseCurve:
    step: tuple[int, ...]
    value: tuple[float, ...]


@dataclass(kw_only=True)
class CasePanel:
    title: str
    metric: str
    arms: dict[str, CaseCurve]
    threshold: float
    kind: str = "case"
    task_id: str | None = None
    checkpoint: int | None = None
    start_excess: float | None = None


Panel = PolicyPanel | BridgePanel | CasePanel


@dataclass(kw_only=True)
class Thresholds:
    separation: float
    excess_concordance: float


@dataclass(kw_only=True)
class PanelSpecifications:
    schema: int
    data_manifest_sha256: str
    completed: int
    planned: int
    core_completed: int
    core_planned: int
    source_dataset: str
    plan_fingerprint: str
    coverage: tuple[CoverageRow, ...]
    panels: dict[str, Panel]
    thresholds: Thresholds
    comparison_unit: str


@dataclass(kw_only=True)
class PlotStyle:
    font_size: float = 8
    marker_size: float = 3.3
    colors: dict[str, str] = field(default_factory=dict)
    heatmap_limit: float = 1.
    bridge_detail_columns: tuple[int, ...] = (0, 1)


@dataclass(kw_only=True)
class FigureLayout:
    width: float
    height: float
    rows: int
    columns: int
    left: float
    right: float
    bottom: float
    top: float
    hspace: float
    wspace: float
    panels: tuple[str, ...]


@dataclass(kw_only=True)
class PlotLayout:
    schema: int
    style: PlotStyle
    figures: dict[str, FigureLayout]


@dataclass(kw_only=True)
class RenderStyle(PlotStyle):
    bridge_limits: dict[str, tuple[float, float]]
    case_limits: dict[str, tuple[float, float]]
    bridge_panel_limits: dict[tuple[str, str], tuple[float, float]]


@dataclass(frozen=True)
class IllustrativeCase:
    name: str
    task_id: str
    stage: str
    title: str


@dataclass(frozen=True)
class CaseLineStyle:
    ls: str | tuple[int, tuple[int, ...]]
    marker: str
    markevery: tuple[int, int]
    lw: float
    zorder: int


def parse_panel(wire: dict) -> Panel:
    kinds = {"policy": PolicyPanel, "bridge": BridgePanel, "case": CasePanel}
    if wire.get("kind") not in kinds:
        raise ValueError("unsupported panel kind")
    return parse_dataclass(kinds[wire["kind"]], wire)


def parse_specifications(wire: dict) -> PanelSpecifications:
    if wire.get("schema") != 1:
        raise ValueError("unsupported panel schema")
    # The union's discriminator is decoded explicitly at the file boundary.
    panels = {name: parse_panel(panel) for name, panel in wire["panels"].items()}
    remaining = {name: value for name, value in wire.items() if name != "panels"}
    definitions = PanelSpecifications.__dataclass_fields__
    if remaining.keys() - definitions.keys():
        raise ValueError("unknown panel specification fields")
    return PanelSpecifications(
        schema=wire["schema"], data_manifest_sha256=wire["data_manifest_sha256"],
        completed=wire["completed"], planned=wire["planned"], core_completed=wire["core_completed"],
        core_planned=wire["core_planned"], source_dataset=wire["source_dataset"],
        plan_fingerprint=wire["plan_fingerprint"],
        coverage=tuple(parse_dataclass(CoverageRow, row) for row in wire["coverage"]),
        panels=panels, thresholds=parse_dataclass(Thresholds, wire["thresholds"]),
        comparison_unit=wire["comparison_unit"],
    )


def parse_layout(wire: dict) -> PlotLayout:
    if wire.get("schema") != 1:
        raise ValueError("unsupported layout schema")
    return parse_dataclass(PlotLayout, wire)
