

# Stage 2: Build backend
FROM python:3.11-slim as backend-build

WORKDIR /app/backend

# Copy requirements and install dependencies
COPY backend/requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the backend source code
COPY backend/ .

# Stage 3: Final image
FROM python:3.11-slim

# Set the working directory for the final stage
WORKDIR /app


# Copy backend files from the backend build stage
COPY --from=backend-build /app/backend ./backend

# Explicitly copy HandTrackingModule.py to the root directory
COPY --from=backend-build /app/backend/HandTrackingModule.py ./

# Explicitly copy the assets folder to the root directory
COPY --from=backend-build /app/backend/assets ./assets

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Reinstall requirements in the final stage (redundant files removed from previous stage)
COPY backend/requirements.txt ./backend/
RUN pip install --upgrade pip && pip install --no-cache-dir -r backend/requirements.txt

# Install Python dependencies directly
RUN pip install --upgrade pip && \
    pip install fastapi uvicorn[standard] websocket

# Expose the port FastAPI will run on
EXPOSE 7860

# Command to run the FastAPI app
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]