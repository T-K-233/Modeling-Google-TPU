import unittest
from pathlib import Path

from tpu.parser import BundleParser


class ParserSlotAssignmentTests(unittest.TestCase):
    def setUp(self):
        self.parser = BundleParser()

    def test_scalar_overflow_stays_in_scalar_family(self):
        _, bundles = self.parser.parse_program(
            Path("tests/vector_add_f32/tpu_compiler_dump/llo/1772770869978023816-add.3")
        )
        bundle = bundles[0]

        self.assertFalse(bundle.mxu0.valid)
        self.assertFalse(bundle.xlu0.valid)
        self.assertFalse(bundle.valu0.valid)
        self.assertFalse(bundle.load0.valid)
        self.assertFalse(bundle.store0.valid)

        self.assertEqual(bundle.salu0.opcode, "vsyncpa")
        self.assertEqual(bundle.salu1.opcode, "inlined_call_operand.hbm")
        self.assertEqual(
            [slot.opcode for slot in bundle.overflow_slots],
            ["inlined_call_operand.hbm", "inlined_call_operand.hbm"],
        )
        self.assertEqual(
            [slot.opcode for slot in bundle.iter_valid_slots()],
            [
                "vsyncpa",
                "inlined_call_operand.hbm",
                "inlined_call_operand.hbm",
                "inlined_call_operand.hbm",
            ],
        )
        self.assertEqual(bundle.table_repr().count("inlined_call_operand.hbm"), 3)

    def test_mixed_bundle_keeps_original_slot_families(self):
        _, bundles = self.parser.parse_program(
            Path("tests/linear_f32/tpu_compiler_dump/llo/1772821895612304246-reshape.2")
        )
        bundle = bundles[0x5]

        self.assertEqual(bundle.xlu1.opcode, "vrot.lane.b32.xlu1")
        self.assertEqual(bundle.xlu0.opcode, "vrot.lane.b32.xlu0")
        self.assertEqual(bundle.store0.opcode, "vst.msk")
        self.assertEqual(bundle.salu0.opcode, "smov")
        self.assertEqual(bundle.salu1.opcode, "smov")
        self.assertEqual(bundle.overflow_slots, [])


if __name__ == "__main__":
    unittest.main()
