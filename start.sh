#!/bin/bash
# start.sh

# Use Railway's PORT environment variable
export STREAMLIT_SERVER_PORT=$PORT

# If PORT is not set, default to 8080
if [ -z "$PORT" ]; then
    export STREAMLIT_SERVER_PORT=8080
fi

echo "Starting Streamlit on port: $STREAMLIT_SERVER_PORT"

# Start the application
exec streamlit run app.py --server.port=$STREAMLIT_SERVER_PORT --server.address=0.0.0.0
