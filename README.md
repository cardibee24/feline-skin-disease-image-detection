# Feline Skin Disease Image Detection App

This application uses **React (Vite)** for the frontend and **FastAPI** for the backend to detect feline skin diseases from images.

## Prerequisites

1.  **Node.js**: [Download & Install](https://nodejs.org/)
2.  **Python** (3.8+): [Download & Install](https://www.python.org/)
    - **Important**: During installation, check "Add Python to PATH".

## Setup & Run

### 1. Backend (API & Model)

The backend handles image processing and ML inference.

1.  Open a terminal in the `backend` folder:
    ```bash
    cd backend
    ```
2.  Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    .\venv\Scripts\Activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Training (Optional)**:
    - Place your images in `backend/dataset/train/<disease_name>` and `backend/dataset/val/<disease_name>`.
    - Run the training script:
      ```bash
      python train.py
      ```
    - This will generate `model.pth`.
5.  **Start the Server**:
    ```bash
    python main.py
    ```
    - The API will run at `http://localhost:8000`.

### 2. Frontend (User Interface)

The frontend allows you to upload images and view results.

1.  Open a new terminal in the `frontend` folder:
    ```bash
    cd frontend
    ```
2.  Install dependencies (if not already done):
    ```bash
    npm install
    ```
3.  Start the development server:
    ```bash
    npm run dev
    ```
4.  Open your browser at `http://localhost:5173`.

## Usage

1.  Ensure the **Backend** is running.
2.  Go to the **Frontend** URL.
3.  Upload a cat image.
4.  View the prediction result!
    - **Note**: If you haven't trained a model, the app will use a **Mock Model** which returns random results for demonstration.
