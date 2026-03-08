from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass(slots=True)
class AsqSettings:
    enabled: bool = True
    min_hits: int = 3
    include_transfers: bool = False
    max_amount_diff: float = 2500.0
    period_days: int = 30

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AsqSettings:
        data = data or {}
        return cls(
            enabled=_bool(data.get("enabled", True)),
            min_hits=int(data.get("min_hits", 3)),
            include_transfers=_bool(data.get("include_transfers", False)),
            max_amount_diff=float(data.get("max_amount_diff", 2500.0)),
            period_days=max(1, int(data.get("period_days", 30))),
        )


@dataclass(slots=True)
class GlobalSettings:
    r_cycle: int = 14
    r_cost: float = 8.0
    k_cost: float = 0.18
    stockout_penalty: float = 5.0
    expedite_rate: float = 0.03
    gm: float = 0.15
    realization: float = 1.0
    auto_po_enabled: bool = False
    asq: AsqSettings = field(default_factory=AsqSettings)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GlobalSettings:
        data = data or {}
        return cls(
            r_cycle=int(data.get("r_cycle", 14)),
            r_cost=float(data.get("r_cost", 8.0)),
            k_cost=float(data.get("k_cost", 0.18)),
            stockout_penalty=float(data.get("stockout_penalty", 5.0)),
            expedite_rate=float(data.get("expedite_rate", 0.03)),
            gm=float(data.get("gm", 0.15)),
            realization=float(data.get("realization", 1.0)),
            auto_po_enabled=_bool(data.get("auto_po_enabled", False)),
            asq=AsqSettings.from_dict(data.get("asq")),
        )


@dataclass(slots=True)
class Receipt:
    receipt_id: str
    qty: float
    eta_day: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Receipt:
        return cls(
            receipt_id=str(data.get("receipt_id", data.get("id", ""))),
            qty=float(data.get("qty", 0.0)),
            eta_day=int(data.get("eta_day", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.receipt_id,
            "receipt_id": self.receipt_id,
            "qty": self.qty,
            "eta_day": self.eta_day,
        }


@dataclass(slots=True)
class ServiceMetrics:
    orders: int = 0
    orders_stockout: int = 0
    units_ordered: float = 0.0
    units_shipped: float = 0.0
    units_backordered: float = 0.0
    zero_on_hand_hits: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ServiceMetrics:
        data = data or {}
        return cls(
            orders=int(data.get("orders", 0)),
            orders_stockout=int(data.get("orders_stockout", 0)),
            units_ordered=float(data.get("units_ordered", 0.0)),
            units_shipped=float(data.get("units_shipped", 0.0)),
            units_backordered=float(data.get("units_backordered", 0.0)),
            zero_on_hand_hits=int(data.get("zero_on_hand_hits", 0)),
        )


@dataclass(slots=True)
class CostMetrics:
    ordering: float = 0.0
    holding: float = 0.0
    stockout: float = 0.0
    expedite: float = 0.0
    purchases: float = 0.0
    total: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CostMetrics:
        data = data or {}
        return cls(
            ordering=float(data.get("ordering", 0.0)),
            holding=float(data.get("holding", 0.0)),
            stockout=float(data.get("stockout", 0.0)),
            expedite=float(data.get("expedite", 0.0)),
            purchases=float(data.get("purchases", 0.0)),
            total=float(data.get("total", 0.0)),
        )


@dataclass(slots=True)
class SalesMetrics:
    revenue: float = 0.0
    cogs: float = 0.0
    units_sold: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SalesMetrics:
        data = data or {}
        return cls(
            revenue=float(data.get("revenue", 0.0)),
            cogs=float(data.get("cogs", 0.0)),
            units_sold=float(data.get("units_sold", 0.0)),
        )


@dataclass(slots=True)
class AnalyticsMetrics:
    inv_value_daysum: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AnalyticsMetrics:
        data = data or {}
        return cls(inv_value_daysum=float(data.get("inv_value_daysum", 0.0)))


@dataclass(slots=True)
class ExceptionRecord:
    day: int
    item_index: int
    code: str
    message: str
    op: float
    op_base_raw: float
    asq: float
    item_cost: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExceptionRecord:
        return cls(
            day=int(data.get("day", 0)),
            item_index=int(data.get("item_index", 0)),
            code=str(data.get("code", "")),
            message=str(data.get("message", "")),
            op=float(data.get("op", 0.0)),
            op_base_raw=float(data.get("op_base_raw", 0.0)),
            asq=float(data.get("asq", 0.0)),
            item_cost=float(data.get("item_cost", 0.0)),
        )


@dataclass(slots=True)
class InventoryItem:
    usage_rate: float
    lead_time: float
    item_cost: float
    safety_allowance: float
    standard_pack: float
    hits_per_month: float
    daily_ur: float = 0.0
    op: float = 0.0
    lp: float = 0.0
    eoq: float = 0.0
    oq: float = 0.0
    surplus_line: float = 0.0
    cp: float = 0.0
    on_hand: float = 0.0
    pipeline: list[Receipt] = field(default_factory=list)
    backorder: float = 0.0
    stockout_today: bool = False
    asq_hits_period: int = 0
    asq_usage_period: float = 0.0
    asq_last_value: float = 0.0
    asq_last_applied_day: int = 0
    op_base_raw: float = 0.0
    pna: float = 0.0
    pna_days: float = 0.0
    pna_days_frm_op: float = 0.0
    soq: float = 0.0
    proposed_pna: float = 0.0
    pro_pna_days_frm_op: float = 0.0
    no_pna_days_frm_op: float = 0.0
    ats_days_frm_op: float = 0.0
    ats_days_to_stockout: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InventoryItem:
        safety_allowance = data.get(
            "safety_allowance",
            float(data.get("safety_allowance_pct", 0.0)) / 100.0,
        )
        return cls(
            usage_rate=float(data.get("usage_rate", 0.0)),
            lead_time=float(data.get("lead_time", data.get("lead_time_days", 0.0))),
            item_cost=float(data.get("item_cost", 0.0)),
            safety_allowance=float(safety_allowance),
            standard_pack=max(1.0, float(data.get("standard_pack", 1.0))),
            hits_per_month=max(0.01, float(data.get("hits_per_month", 1.0))),
            daily_ur=float(data.get("daily_ur", 0.0)),
            op=float(data.get("op", 0.0)),
            lp=float(data.get("lp", 0.0)),
            eoq=float(data.get("eoq", 0.0)),
            oq=float(data.get("oq", 0.0)),
            surplus_line=float(data.get("surplus_line", 0.0)),
            cp=float(data.get("cp", 0.0)),
            on_hand=float(data.get("on_hand", data.get("pna", 0.0))),
            pipeline=[Receipt.from_dict(rec) for rec in data.get("pipeline", [])],
            backorder=float(data.get("backorder", 0.0)),
            stockout_today=_bool(data.get("stockout_today", False)),
            asq_hits_period=int(data.get("asq_hits_period", 0)),
            asq_usage_period=float(data.get("asq_usage_period", 0.0)),
            asq_last_value=float(data.get("asq_last_value", 0.0)),
            asq_last_applied_day=int(data.get("asq_last_applied_day", 0)),
            op_base_raw=float(data.get("op_base_raw", data.get("op", 0.0))),
            pna=float(data.get("pna", 0.0)),
            pna_days=float(data.get("pna_days", 0.0)),
            pna_days_frm_op=float(data.get("pna_days_frm_op", 0.0)),
            soq=float(data.get("soq", 0.0)),
            proposed_pna=float(data.get("proposed_pna", 0.0)),
            pro_pna_days_frm_op=float(data.get("pro_pna_days_frm_op", 0.0)),
            no_pna_days_frm_op=float(data.get("no_pna_days_frm_op", 0.0)),
            ats_days_frm_op=float(data.get("ats_days_frm_op", 0.0)),
            ats_days_to_stockout=float(data.get("ats_days_to_stockout", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["pipeline"] = [receipt.to_dict() for receipt in self.pipeline]
        data["lead_time_days"] = self.lead_time
        data["safety_allowance_pct"] = self.safety_allowance * 100.0
        return data


@dataclass(slots=True)
class TrainingProfile:
    current_view: str = "main_menu"
    active_level_id: str | None = None
    highest_unlocked_level: int = 1
    completed_levels: list[str] = field(default_factory=list)
    simulator_unlocked: bool = False
    auto_po_reward_unlocked: bool = False
    lesson_status: str = "idle"
    last_result_title: str = ""
    last_result_message: str = ""
    guided_orders_placed: int = 0
    custom_orders_placed: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TrainingProfile:
        data = data or {}
        completed_levels = [str(level_id) for level_id in data.get("completed_levels", [])]
        return cls(
            current_view=str(data.get("current_view", "main_menu")),
            active_level_id=(
                None
                if data.get("active_level_id") in (None, "")
                else str(data.get("active_level_id"))
            ),
            highest_unlocked_level=max(1, int(data.get("highest_unlocked_level", 1))),
            completed_levels=completed_levels,
            simulator_unlocked=_bool(data.get("simulator_unlocked", False)),
            auto_po_reward_unlocked=_bool(data.get("auto_po_reward_unlocked", False)),
            lesson_status=str(data.get("lesson_status", "idle")),
            last_result_title=str(data.get("last_result_title", "")),
            last_result_message=str(data.get("last_result_message", "")),
            guided_orders_placed=max(0, int(data.get("guided_orders_placed", 0))),
            custom_orders_placed=max(0, int(data.get("custom_orders_placed", 0))),
        )


@dataclass(slots=True)
class SimulationState:
    global_settings: GlobalSettings = field(default_factory=GlobalSettings)
    items: list[InventoryItem] = field(default_factory=list)
    day: int = 1
    is_initialized: bool = False
    service_today: ServiceMetrics = field(default_factory=ServiceMetrics)
    service_totals: ServiceMetrics = field(default_factory=ServiceMetrics)
    costs: CostMetrics = field(default_factory=CostMetrics)
    sales: SalesMetrics = field(default_factory=SalesMetrics)
    exception_center: list[ExceptionRecord] = field(default_factory=list)
    analytics: AnalyticsMetrics = field(default_factory=AnalyticsMetrics)
    training: TrainingProfile = field(default_factory=TrainingProfile)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SimulationState:
        data = deepcopy(data or {})
        return cls(
            global_settings=GlobalSettings.from_dict(data.get("global_settings")),
            items=[InventoryItem.from_dict(item) for item in data.get("items", [])],
            day=int(data.get("day", 1)),
            is_initialized=_bool(data.get("is_initialized", False)),
            service_today=ServiceMetrics.from_dict(data.get("service_today")),
            service_totals=ServiceMetrics.from_dict(data.get("service_totals")),
            costs=CostMetrics.from_dict(data.get("costs")),
            sales=SalesMetrics.from_dict(data.get("sales")),
            exception_center=[
                ExceptionRecord.from_dict(rec) for rec in data.get("exception_center", [])
            ],
            analytics=AnalyticsMetrics.from_dict(data.get("analytics")),
            training=TrainingProfile.from_dict(data.get("training")),
        )

    def clone(self) -> SimulationState:
        return SimulationState.from_dict(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["items"] = [item.to_dict() for item in self.items]
        return data


def default_state() -> SimulationState:
    return SimulationState()
