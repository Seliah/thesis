# Command to run prototype in development mode
uvicorn analysis.api:app \
  --reload \
  --host 0.0.0.0 \
  --port 8450 \
  --log-level debug
