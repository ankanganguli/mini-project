# 08_UI_SPEC.md: Frontend Design & UI Specification

## 1. Design Tokens (Starbucks Modern Light Palette)
- **Canvas Background:** `#F9F6F0` (Warm Cream Canvas)
- **Surface Cards:** `#FFFFFF` (Crisp White with subtle shadow)
- **Primary / Starbucks Accent:** `#006241` (Starbucks Signature Green)
- **Deep Forest Text:** `#1E3932` (High contrast typography)
- **Border Tint / Subtle Accents:** `#D4E9E2` (Soft Mint Green)

## 2. Interactive 3D Hero Element
- Built with **Three.js (r128)** rendering a floating geometric forensic icosahedron.
- **Idle State:** Responds organically to mouse tilt and cursor parallax with subtle wireframe and vertex particle glow.
- **Verification Active State:** Shifts dynamically into an accelerated green pulse rotation when an API verification request is dispatched, providing instant physical feedback.

## 3. Visual Components & Layout
- **Radial SVG Gauge:** Animating score confidence smoothly from 0% to 100% with color transitioning according to authenticity status.
- **Multimodal Claim Input:** Dual tab / side-by-side card for text claim input and drag-and-drop media upload (supporting JPEG, PNG, MP4, WebM).
- **Forensic Breakdown Card:** Visualizes text calibrated probability, ELA visual entropy, and weighted fusion formula.
- **Audit History Feed:** Real-time log showing latest verifications fetched from the MongoDB-backed API.
