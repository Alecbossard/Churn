# Path Planning Studio

Interactive 3D platform for prototyping and evaluating autonomous racing path-planning algorithms before their integration into the **ROS 2 / Gazebo Formula Student stack**.

Developed in the context of **TLSe Racing**, the platform provides a fast way to test racing-line generation, trajectory optimisation and vehicle behaviour on cone-defined tracks without requiring the full ROS 2 simulation environment.

## Highlights

- Interactive cone-based track editor
- Real-time 3D visualisation with multiple camera modes
- RRT*, QP, Laplacian and hybrid trajectory optimisation
- Local planning mode for short-horizon experiments
- Vehicle simulation with velocity, acceleration and braking
- Ghost-car comparison
- Live G-G diagram and telemetry
- Online deployment on Google Cloud Run

## Live Demo

**Online version:**  
https://path-planning-475644098248.us-west1.run.app/

The web application can be used directly in the browser without any local installation.

---

## Demo

### RRT* + QP Racing Line

![RRT* + QP racing line](docs/demo-rrtqp.gif)

### Platform Overview

![Path Planning Studio – site overview](docs/demo-site.gif)

---

## Motivation

Before integrating path-planning algorithms into the full ROS 2 / Gazebo Formula Student environment, I wanted a lightweight tool for quickly visualising and comparing different trajectory-generation strategies.

Path Planning Studio was therefore developed as a rapid prototyping environment to:

- import or edit cone-defined racing tracks,
- generate an initial centerline,
- test multiple racing-line optimisation approaches,
- compare trajectories visually,
- simulate vehicle motion,
- analyse curvature, velocity and lateral/longitudinal acceleration.

The selected approaches could then be transferred and evaluated in the TLSe Racing ROS 2 simulation stack.

---

## Features

### Cone-Based Track Editor

- Import tracks from CSV files containing blue, yellow and orange cones.
- Support for start/finish and vehicle-start cones.
- Interactive track editing.
- Automatic centerline generation from track boundaries.
- Dynamic track-width handling.

### 3D Visualisation

Built with `@react-three/fiber` and `three.js`.

Available views include:

- Orbit camera
- Chase camera
- Cockpit camera
- Helicopter camera

The environment also includes:

- day/night visualisation,
- cone meshes,
- vehicle model,
- trajectory overlays,
- track boundaries.

### Trajectory Generation

The platform converts the cone-defined circuit into a dense path representation containing:

- position,
- curvature,
- arc length,
- track-width information.

The trajectory is automatically recomputed when the track geometry is modified.

### Racing-Line Optimisation

The optimisation methods are implemented in `services/mathUtils.ts`.

Available approaches include:

| Method | Description |
|---|---|
| **Laplacian** | Smooths the centerline using Laplacian filtering |
| **RRT*** | Searches for shorter valid shortcuts within the track boundaries |
| **QP** | Minimum-curvature optimisation using biharmonic smoothing |
| **Hybrid** | Combines QP and Laplacian smoothing |
| **RRT* + QP** | Uses RRT* for shortcut generation followed by QP smoothing |
| **Local** | Short-horizon planner operating around the current vehicle position |

These methods allow quick visual comparison between trajectory smoothness, length and curvature.

---

## Vehicle Simulation & Telemetry

The platform includes a lightweight vehicle simulation to evaluate the generated trajectories.

### Vehicle Behaviour

- Trajectory following
- Longitudinal velocity model
- Acceleration and braking
- Curvature-dependent behaviour

### Ghost Comparison

A ghost vehicle can be enabled to compare the current trajectory with a precomputed reference trajectory.

### Telemetry

The interface provides:

- vehicle velocity,
- longitudinal acceleration,
- lateral acceleration,
- g-force evolution,
- live G-G diagram,
- trajectory visualisation.

Charts are rendered using **Recharts**.

---

## Tech Stack

### Frontend

- React
- TypeScript
- Vite

### 3D

- Three.js
- React Three Fiber
- React Three Drei

### Visualisation

- Recharts

### Algorithms

- RRT*
- Quadratic Programming
- Laplacian smoothing
- Minimum-curvature optimisation
- Local path planning

### Deployment

- Google Cloud Run

---

## Project Structure

```text
Path-Planning-Studio/
├── App.tsx
├── index.tsx
├── constants.ts
├── types.ts
├── components/
│   ├── AlgorithmsPage.tsx
│   ├── Car.tsx
│   ├── LandingPage.tsx
│   ├── Scene3D.tsx
│   ├── SimulationsPage.tsx
│   ├── TrackObjects.tsx
│   ├── UIOverlay.tsx
│   └── WorldEnvironment.tsx
├── services/
│   └── mathUtils.ts
├── docs/
│   ├── demo-rrtqp.gif
│   └── demo-site.gif
├── package.json
├── tsconfig.json
└── vite.config.ts
```

### Main Components

- `Scene3D.tsx` — 3D environment, track, vehicle and trajectories
- `Car.tsx` — vehicle model and motion
- `TrackObjects.tsx` — cone and track geometry
- `UIOverlay.tsx` — telemetry, controls and HUD
- `AlgorithmsPage.tsx` — explanation and selection of optimisation methods
- `SimulationsPage.tsx` — simulation scenarios
- `mathUtils.ts` — centerline generation and trajectory-optimisation algorithms

---

## Getting Started

### Prerequisites

- Node.js 20+
- npm

### Installation

```bash
git clone https://github.com/Alecbossard/Path-Planning-Studio.git
cd Path-Planning-Studio
npm install
```

### Run Locally

```bash
npm run dev
```

Vite will start the development server, typically at:

```text
http://localhost:5173
```

### Build

```bash
npm run build
```

---

## Typical Workflow

1. Open the online platform or run it locally.
2. Select or import a cone-based track.
3. Modify the circuit if needed.
4. Generate the centerline.
5. Select a trajectory optimiser.
6. Compare the generated racing lines.
7. Launch the vehicle simulation.
8. Analyse:
   - trajectory geometry,
   - velocity,
   - curvature,
   - lateral and longitudinal acceleration,
   - G-G diagram.
9. Use the results to select approaches for further testing in the ROS 2 / Gazebo Formula Student environment.

---

## Context

This project was developed as part of my work on autonomous path planning with **TLSe Racing**.

The web platform was used as an intermediate experimentation layer before moving selected algorithms into the ROS 2 Formula Student simulation environment.

Related topics:

`Autonomous Driving` · `Path Planning` · `Motion Planning` · `Trajectory Optimisation` · `Formula Student` · `ROS 2`

---

## Author

**Alec Bossard**  
Engineering student in Robotics and Interactive Systems at UPSSITECH – University of Toulouse.

[LinkedIn](https://www.linkedin.com/in/alec-bossard/)
