from dataclasses import dataclass

RISK_ORDER = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

PRODUCT_PROMPTS = [
    "cardboard box", "carton", "package", "parcel", "crate", "container",
    "mattress", "chair", "sofa", "table", "appliance", "furniture", "bag",
    "pallet", "wooden pallet", "forklift", "pallet jack", "hand truck",
    "dolly", "cart", "trolley", "person", "worker", "operator",
]

PERSON_NAMES = {"person", "worker", "operator", "employee", "human"}
EQUIPMENT_NAMES = {"pallet", "wooden pallet", "forklift", "pallet jack", "hand truck", "dolly", "cart", "trolley"}

SCENARIOS = {
    "Dragging": ("Product is pulled across a surface instead of being carried or transported correctly.", "HIGH", "Use approved handling equipment and avoid pulling product directly across the floor."),
    "Rolling": ("A product is moved by uncontrolled rolling rather than the intended handling method.", "HIGH", "Stop uncontrolled rolling and reposition the product using approved handling equipment."),
    "Throwing/Dropping": ("A product shows abrupt release or rapid downward motion consistent with a drop or throw.", "CRITICAL", "Pause handling, inspect the evidence and verify the product before continuing."),
    "Improper stacking": ("Tracked products form a geometrically unstable or poorly supported stack.", "HIGH", "Re-stack products with stable support and sufficient overlap."),
    "Stepping on cartons": ("A person's foot region overlaps the top surface of a tracked product.", "CRITICAL", "Stop the activity and keep people off cartons and packages."),
    "Heavy item on light/fragile item": ("A visually larger product is persistently positioned above a substantially smaller product.", "HIGH", "Place larger/heavier products on suitable lower support positions."),
    "Vertical product kept horizontally": ("A product's observed bounding-box orientation remains substantially flatter than its earlier orientation.", "MEDIUM", "Review the product orientation and restore the intended upright position if required."),
    "Outside designated area": ("A tracked product crossed outside the configured operating zone.", "MEDIUM", "Review the configured zone and move the product back into the designated area."),
    "Unattended product": ("A product remained nearly stationary for a sustained period without a nearby person.", "MEDIUM", "Confirm the product is safely staged or assign an operator to attend to it."),
    "Unsafe loading/unloading sequence": ("A new product entered the configured loading zone before an earlier product had settled.", "HIGH", "Complete one placement before introducing the next product into the active loading zone."),
    "Rough handling / excessive force": ("A tracked product shows an abrupt high-magnitude change in motion.", "HIGH", "Slow the handling action and use controlled movement."),
}

@dataclass(frozen=True)
class Zone:
    name: str
    x1: float
    y1: float
    x2: float
    y2: float

DEFAULT_ZONES = {
    "loading_zone": Zone("Loading zone", 0.20, 0.20, 0.80, 0.90),
    "designated_zone": Zone("Designated handling area", 0.05, 0.10, 0.95, 0.95),
}


def risk_color(risk: str) -> str:
    return {"LOW": "#45d6a8", "MEDIUM": "#f4c95d", "HIGH": "#ff984d", "CRITICAL": "#ff4d6d"}.get(risk, "#9aa9b2")
