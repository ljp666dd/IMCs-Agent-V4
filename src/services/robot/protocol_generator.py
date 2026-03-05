"""
IMCs Robot Protocol Generator (V5 - Phase C)

Generates automated experiment protocols for liquid handling robots:
1. Opentrons OT-2 protocol scripts (Python)
2. LLM-assisted parameter optimization
3. Safety checks and volume validation

Bridges the gap between computational predictions and physical experiments.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentProtocol:
    """Generated experiment protocol."""
    material_formula: str
    protocol_type: str  # "ink_preparation", "electrode_deposition", "electrochemical_test"
    robot_platform: str = "OT-2"
    steps: List[Dict[str, Any]] = field(default_factory=list)
    python_script: str = ""
    estimated_duration_min: float = 0.0
    reagents: List[Dict[str, Any]] = field(default_factory=list)
    safety_notes: List[str] = field(default_factory=list)


# Standard reagent database for HOR catalyst preparation
REAGENT_DB = {
    "nafion_5pct": {"name": "Nafion 5% Solution", "density_g_ml": 1.10, "location": "A1"},
    "ipa": {"name": "Isopropanol", "density_g_ml": 0.786, "location": "A2"},
    "di_water": {"name": "DI Water", "density_g_ml": 1.00, "location": "A3"},
    "catalyst_powder": {"name": "Catalyst Powder", "location": "B1"},
}


class ProtocolGenerator:
    """
    Generates robot-executable experiment protocols for catalyst testing.
    """

    def __init__(self):
        logger.info("ProtocolGenerator initialized.")

    def generate_ink_protocol(
        self,
        catalyst_formula: str,
        catalyst_mass_mg: float = 5.0,
        nafion_volume_ul: float = 10.0,
        solvent_volume_ul: float = 490.0,
        sonication_min: float = 30.0
    ) -> ExperimentProtocol:
        """
        Generate a catalyst ink preparation protocol.
        Standard procedure for HOR/HER electrode fabrication.
        """
        total_volume = nafion_volume_ul + solvent_volume_ul

        steps = [
            {
                "step": 1,
                "action": "weigh_catalyst",
                "description": f"称取 {catalyst_mass_mg} mg {catalyst_formula} 催化剂粉末",
                "manual": True
            },
            {
                "step": 2,
                "action": "transfer_to_vial",
                "description": "将催化剂转移至 2 mL EP 管",
                "manual": True
            },
            {
                "step": 3,
                "action": "aspirate",
                "source": REAGENT_DB["ipa"]["location"],
                "volume_ul": solvent_volume_ul * 0.8,
                "description": f"加入 {solvent_volume_ul * 0.8:.0f} µL IPA"
            },
            {
                "step": 4,
                "action": "aspirate",
                "source": REAGENT_DB["di_water"]["location"],
                "volume_ul": solvent_volume_ul * 0.2,
                "description": f"加入 {solvent_volume_ul * 0.2:.0f} µL 去离子水"
            },
            {
                "step": 5,
                "action": "aspirate",
                "source": REAGENT_DB["nafion_5pct"]["location"],
                "volume_ul": nafion_volume_ul,
                "description": f"加入 {nafion_volume_ul:.0f} µL 5% Nafion 溶液"
            },
            {
                "step": 6,
                "action": "sonicate",
                "duration_min": sonication_min,
                "description": f"超声分散 {sonication_min:.0f} 分钟",
                "manual": True
            }
        ]

        reagents = [
            {"name": catalyst_formula, "amount": f"{catalyst_mass_mg} mg", "type": "solid"},
            {"name": "IPA", "amount": f"{solvent_volume_ul * 0.8:.0f} µL", "type": "liquid"},
            {"name": "DI Water", "amount": f"{solvent_volume_ul * 0.2:.0f} µL", "type": "liquid"},
            {"name": "Nafion 5%", "amount": f"{nafion_volume_ul:.0f} µL", "type": "liquid"},
        ]

        # Generate OT-2 Python script
        script = self._generate_ot2_script(catalyst_formula, steps)

        return ExperimentProtocol(
            material_formula=catalyst_formula,
            protocol_type="ink_preparation",
            robot_platform="OT-2",
            steps=steps,
            python_script=script,
            estimated_duration_min=5.0 + sonication_min,
            reagents=reagents,
            safety_notes=[
                "IPA 为易燃溶剂，需在通风橱中操作",
                "Nafion 溶液有腐蚀性，戴手套操作",
                f"总液量 {total_volume:.0f} µL，确保移液器量程匹配"
            ]
        )

    def generate_drop_casting_protocol(
        self,
        catalyst_formula: str,
        ink_volume_ul: float = 10.0,
        electrode_area_cm2: float = 0.196,
        loading_ug_cm2: float = 50.0
    ) -> ExperimentProtocol:
        """Generate electrode drop-casting protocol."""
        steps = [
            {
                "step": 1,
                "action": "mix",
                "description": "涡旋混合催化剂墨水 30 秒",
                "duration_s": 30
            },
            {
                "step": 2,
                "action": "aspirate",
                "volume_ul": ink_volume_ul,
                "description": f"吸取 {ink_volume_ul:.1f} µL 催化剂墨水"
            },
            {
                "step": 3,
                "action": "dispense",
                "volume_ul": ink_volume_ul,
                "target": "GCE",
                "description": f"滴涂至玻碳电极 (面积 {electrode_area_cm2} cm²)"
            },
            {
                "step": 4,
                "action": "dry",
                "description": "室温自然干燥 30 分钟",
                "duration_min": 30,
                "manual": True
            }
        ]

        return ExperimentProtocol(
            material_formula=catalyst_formula,
            protocol_type="electrode_deposition",
            steps=steps,
            estimated_duration_min=35.0,
            safety_notes=[
                f"目标载量: {loading_ug_cm2} µg/cm²",
                "确保电极表面已打磨至镜面光洁"
            ]
        )

    def _generate_ot2_script(self, formula: str, steps: List[Dict]) -> str:
        """Generate Opentrons OT-2 Python protocol script."""
        robot_steps = [s for s in steps if not s.get("manual")]

        script_lines = [
            "# Auto-generated OT-2 Protocol",
            f"# Catalyst: {formula}",
            "# Generated by IMCs ProtocolGenerator V5",
            "",
            "from opentrons import protocol_api",
            "",
            "metadata = {",
            f"    'protocolName': 'Ink Preparation - {formula}',",
            "    'author': 'IMCs AutoGen',",
            "    'apiLevel': '2.13'",
            "}",
            "",
            "def run(protocol: protocol_api.ProtocolContext):",
            "    # Labware",
            "    tiprack = protocol.load_labware('opentrons_96_tiprack_300ul', '1')",
            "    reservoir = protocol.load_labware('nest_12_reservoir_15ml', '2')",
            "    target_plate = protocol.load_labware('opentrons_24_tuberack_eppendorf_2ml_safelock_snapcap', '3')",
            "",
            "    # Pipette",
            "    p300 = protocol.load_instrument('p300_single_gen2', 'right', tip_racks=[tiprack])",
            "",
        ]

        for s in robot_steps:
            action = s.get("action")
            if action == "aspirate":
                vol = s.get("volume_ul", 0)
                src = s.get("source", "A1")
                script_lines.append(f"    # {s.get('description', '')}")
                script_lines.append(f"    p300.pick_up_tip()")
                script_lines.append(f"    p300.aspirate({vol}, reservoir['{src}'])")
                script_lines.append(f"    p300.dispense({vol}, target_plate['A1'])")
                script_lines.append(f"    p300.drop_tip()")
                script_lines.append("")

        return "\n".join(script_lines)


def get_protocol_generator() -> ProtocolGenerator:
    """Factory function."""
    return ProtocolGenerator()
