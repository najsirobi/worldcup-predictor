# Annexe C QA Report (Claude)

Generated: 2026-06-04
Repo root: /Users/borisjannijssen/Documents/worldcup-predictor

---

## 1. File Existence

- FOUND   data/reference/third_place_assignment_annex_c.csv
- FOUND   data/reference/round_of_32_mapping.csv
- FOUND   data/reference/knockout_bracket_mapping.csv
- FOUND   data/reference/knockout_round_progression.csv

---

## 2. Annexe C Table QA

- Rows loaded: 495
- Columns: ['option_number', 'qualified_third_groups', 'source', 'notes', 'slot_1A', 'slot_1B', 'slot_1D', 'slot_1E', 'slot_1G', 'slot_1I', 'slot_1K', 'slot_1L']

  [PASS] R1 row count: 495 == 495
  [PASS] R2 all required columns present
  [PASS] R3a option_number is unique
  [PASS] R3b qualified_third_groups is unique
  [PASS] R4 all group letters within A-L
  [PASS] R5 all qualified_third_groups list 8 distinct valid groups
  [PASS] R6 all rows: slot assignments exactly cover qualified groups
  [PASS] R7 no duplicate slot assignments within any row
  [PASS] R8 all 495 expected combinations present

  --- 5 valid row examples ---
  option=1 groups=E,F,G,H,I,J,K,L | slot_1A=3E, slot_1B=3J, slot_1D=3I, slot_1E=3F, slot_1G=3H, slot_1I=3G, slot_1K=3L, slot_1L=3K
  option=2 groups=D,F,G,H,I,J,K,L | slot_1A=3H, slot_1B=3G, slot_1D=3I, slot_1E=3D, slot_1G=3J, slot_1I=3F, slot_1K=3L, slot_1L=3K
  option=3 groups=D,E,G,H,I,J,K,L | slot_1A=3E, slot_1B=3J, slot_1D=3I, slot_1E=3D, slot_1G=3H, slot_1I=3G, slot_1K=3L, slot_1L=3K
  option=4 groups=D,E,F,H,I,J,K,L | slot_1A=3E, slot_1B=3J, slot_1D=3I, slot_1E=3D, slot_1G=3H, slot_1I=3F, slot_1K=3L, slot_1L=3K
  option=5 groups=D,E,F,G,I,J,K,L | slot_1A=3E, slot_1B=3G, slot_1D=3I, slot_1E=3D, slot_1G=3J, slot_1I=3F, slot_1K=3L, slot_1L=3K

---

## 3. R32 Mapping QA

- Rows loaded: 16
  [PASS] Required columns present
  [PASS] R9 R32 has exactly 16 matches
  [PASS] R9 16 distinct R32 match_numbers
  [PASS] R11 exactly 8 best-third slots in R32
  [PASS] R11 all R32 slots have a team source

---

## 4. Round Progression QA

  [PASS] R10 all required rounds present: ['Final', 'QF', 'R16', 'SF']
  [PASS] R10b Final resolves to a tournament Winner

---

## 5. Summary

| File | Status |
|---|---|
| `data/reference/third_place_assignment_annex_c.csv` | Found |
| `data/reference/round_of_32_mapping.csv` | Found |
| `data/reference/knockout_bracket_mapping.csv` | Found |
| `data/reference/knockout_round_progression.csv` | Found |
