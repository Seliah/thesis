# Command to run prototype in production mode
IS_SERVICE=True uvicorn analysis.api:app \
  --host 0.0.0.0 \
  --port 8450 \
  --log-level debug
