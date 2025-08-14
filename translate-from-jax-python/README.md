# Minimal JAX Renderer

A basic differentiable renderer implemented using JAX.

## Features

*   Forward rendering of a simple scene (currently a sphere).
*   Inverse rendering example (parameter estimation).
*   Built with JAX for automatic differentiation.

## Setup

1.  Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Or .\venv\Scripts\activate on Windows
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

*   **Forward Rendering:**
    ```bash
    python forward_render.py
    ```
    This will generate an image `output_render.png` in the project root.

*   **Inverse Rendering:**
    ```bash
    python inverse_render.py
    ```
    This will attempt to optimize material parameters to match a target image. 