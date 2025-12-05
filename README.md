# Design and representation of CMOS fluidic logic; fluidic channel routing and model generation.
Graph of connected nodes => manually editable vector graphics => printable 3D models

# Graph-based Logics Design
yaml graph definition => Inkscape graphics (to implement)
- graph_editor.html: AI generated graph editing tool

# (to do) Information from 2D vector drawing:
1. base body
  - solid body boundary
  - lines to cut from a larger piece
2. fold lines
3. actuator placement/hole across a fold line
4. surface channel lines
  - top
  - bottom
5. internal air channels
6. holes
  - blind
  - through
7. stiffness reduction cut regions
Started: https://github.com/sichengw0826/PneumaticGateInkscape

# 3D Model Generation with Blender Python API (sheet_generation_test.blend):
- Make square hole
- Make circular hole
- Make straight cuts of a pre-defined cross section
- Extrusion along a path that may include curves
- Cut along line segments
- Make "tunnels" of a cross section along multiple segments
