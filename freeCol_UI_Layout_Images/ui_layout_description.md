# freeCol UI Layout — DEFINITIVE UI & LAYOUT SPECIFICATION

Target Framework: PySide6 (Qt Widgets)  
Design Constraint: Must map cleanly to Qt (C++) with no Python-only UI assumptions

---

## GLOBAL UI RULES & CONVENTIONS

- (\*) denotes selected / active UI state
- Top boxes = primary tabs
- Upside-down triangle (▼) = dropdown
- All example list items shown in mockups are illustrative only
- Most lists are EMPTY by default
- Lists generally include Add / Delete controls unless explicitly stated
- All navigation is single-window, non-modal
- Layout resizing occurs only when the main window is resized

---

## 1. PRIMARY NAVIGATION (TOP BAR – PERSISTENT)

Horizontal tab bar at the top of the window:

Initialization | Specifications | Simulation | Results | Modules

- Exactly one active tab
- Switching tabs replaces the entire main content area
- No floating dialogs for navigation

---

## 2. INITIALIZATION TAB

### 2.1 Layout Structure

When Initialization (\*) is active, the content area is split into:

| LHS Vertical Sub-Tab Column | Main Content Pane |

---

### 2.1.1 Secondary Tabs (LHS Column)

Vertical tab list on the left-hand side:

- Thermodynamics (\*)
- Chemical Species

- Only one active at a time
- Selecting a tab swaps the entire right-hand pane

---

## 2.2 THERMODYNAMICS SUB-TAB (\*)

### 2.2.1 Overall Layout

Main content pane is vertically structured:

1. Simulation Model Selection Row
2. Parameter Entry Context Selection
3. Binary Interaction Table Selection
4. Binary Interaction Table Display (conditional)

---

### 2.2.2 Simulation Model Selection (TOP ROW)

Defines which thermodynamic models are actually used in the simulation.

Three dropdowns in a single row:

- Vapor–Liquid Equilibrium ▼
- Activity Coefficient ▼
- Equation of State ▼

Rules:

- These dropdowns define active simulation models
- Changing these does NOT delete parameter data
- Parameter sets are stored per model

---

### 2.2.3 Parameter Entry Context Selection

Controls which model’s parameters are currently being edited.

Context row:

Type: Activity Activity ▼ Model ▼

Rules:

- "Type:" is static text (NOT a dropdown)
- The Activity dropdown may contain:
  - Vapor Pressure
  - Activity
  - EOS
- Model dropdown updates based on Activity selection
- This row ONLY controls parameter-entry context, not simulation behavior

---

### 2.2.4 Binary Interaction Table Selection

A dropdown located directly below the context row.

Purpose: allow editing ONE parameter table at a time.

Example (NRTL selected):

Table ▼ → aij | bij | cij

Rules:

- Dropdown contents depend on selected model
- If the selected model has only one table, this dropdown is hidden
- Changing the selection swaps the visible table in-place

---

### 2.2.5 Binary Interaction Table Display

Rules:

- Tables are only visible if the selected model requires them
- Example:
  - NRTL → tables shown
  - Models without interaction parameters → no table section shown
- Table dimensions: N_species × N_species
- Row and column headers use species names
- Editable numeric cells
- Space allocated for the table is proportionally static
- Table resizes ONLY when the parent window resizes

---

## 2.3 CHEMICAL SPECIES SUB-TAB (\*)

### 2.3.1 Layout

| Species List (LHS) | Species Properties (RHS) |

---

### 2.3.2 Species List Panel

- Vertical list (empty by default)
- Example entries:
  - Species 1
  - Species 2 (\*)
- Buttons:
  - Add
  - Delete

Rules:

- Species are global
- Deleting a species updates:
  - Thermodynamic tables
  - Stream compositions

---

### 2.3.3 Species Properties Panel

- Header: "<Selected Species> Properties"
- Fields depend on selected thermodynamic models
- All fields are numeric and validated
- Properties apply globally to all streams

---

## 3. SPECIFICATIONS TAB

### 3.1 Layout Structure

| LHS Vertical Sub-Tab Column | Main Content Pane |

---

### 3.2 Secondary Tabs (LHS Column)

- Column Config (\*)
- Streams
- Column Overview
- Advanced Modules

---

## 3.3 COLUMN CONFIG SUB-TAB (\*)

### Operating Parameters Section

- Pressure: numeric + units dropdown
- Pressure Drop: numeric (bar/stage)
- Condenser: Configure button
- Reboiler: Configure button

Rules:

- Pressing Configure opens an embedded configuration panel
- Condenser panel shown in mockups is an example only
- Reboiler has a separate but analogous configuration panel
- Panels replace the main content pane (non-modal)

---

## 3.4 STREAMS SUB-TAB (\*)

### Layout

| Stream List (LHS) | Stream Configuration (RHS) |

---

### Stream List Panel

- Empty by default
- Buttons:
  - Add
  - Delete

---

### Stream Configuration Panel

- Type ▼ (Feed, Distillate, Bottoms, Sidestream, etc.)
- Composition: mole fraction per global species
- Stage Number: integer
- Temperature: numeric + units dropdown
- Pressure: numeric + units dropdown

Rules:

- Mole fractions must sum to 1.0
- Species list is auto-generated from global species

---

## 3.5 ADVANCED MODULES SUB-TAB (\*)

### Layout

| Module List (LHS) | Module Configuration (RHS) |

---

### Module List Panel

- Empty by default
- Controls:
  - Add button
  - Dropdown next to Add specifying module type
  - Delete button

Add Dropdown Examples:

- Interreboiler
- Side Stripper
- Side Rectifier

---

### Module Configuration Panel

- Module-specific configuration
- Example shown applies ONLY to Side Stripper

Example (Side Stripper):

- Stage Number: integer
- Number of Stages: integer
- Boilup Ratio (V/B): numeric
- Associated Streams:
  - Distillate (out / to tray)
  - Bottoms (out / to tray)

---

## 4. SIMULATION TAB

### 4.1 Layout Structure

| Solver Method Column (LHS) | Solver Control Pane (RHS) |

---

### 4.2 Solver Method Column (LHS)

- Vertical dropdown:
  - Solver Method ▼
- Options:
  - HYSIM Inside-Out
  - Newton–Raphson
  - etc.

Rules:

- Solver parameters are persisted per solver
- Switching methods restores saved settings

---

### 4.3 Solver Control Pane (RHS)

Split vertically into two rows.

TOP ROW (larger):

- Solver-specific options
- Content generated dynamically per solver

BOTTOM ROW (smaller):

Layout:
| Run / Abort (vertical stack) | Progress Info |

Run / Abort:

- Run button
- Abort button

Progress Info:

- Progress bar
- Iteration count
- Time elapsed

---

## 5. RESULTS TAB

### 5.1 Layout Structure

| LHS Control Column | Results Display Pane |

---

### 5.2 LHS Control Column

Vertical controls in order:

1. Text label: "View"
2. Dropdown ▼: Plot / Table
3. Dropdown ▼:
   - Compositions
   - Temperature
   - Pressure
   - Liquid Flow
   - Vapor Flow
   - etc.

---

### 5.3 Results Display Pane (RHS)

Split vertically.

TOP ROW (larger):

- Plot or table currently selected

BOTTOM ROW (smaller):

- Simulation summary panel
- Read-only status and key outputs

---

## 6. MODULES TAB

### Layout

- Module selection dropdown located in the upper-right corner
- Remaining space used for module-specific UI

---

## FINAL IMPLEMENTATION NOTES FOR THE AGENT

- Use Qt layout managers exclusively (QVBoxLayout, QHBoxLayout, QSplitter)
- Prefer QAbstractItemModel-based lists
- Keep UI and data models separate
- Avoid Python-only shortcuts to ensure C++ portability
