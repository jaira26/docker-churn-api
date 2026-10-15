#!/usr/bin/env bash
# Start a container from the given image, wait until it reports healthy,
# and check that /predict returns a probability. Used locally and in CI.
set -euo pipefail

IMAGE="${1:-churn-api:latest}"
NAME="churn-smoke-test"
PORT="${PORT:-8000}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" -p "$PORT:8000" "$IMAGE" >/dev/null
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT

echo "Waiting for $NAME to become healthy..."
for _ in $(seq 1 30); do
  status="$(docker inspect --format '{{.State.Health.Status}}' "$NAME")"
  [ "$status" = "healthy" ] && break
  sleep 2
done
if [ "$status" != "healthy" ]; then
  echo "Container never became healthy (status: $status)"
  docker logs "$NAME"
  exit 1
fi

response="$(curl -fsS -X POST "http://localhost:$PORT/predict" \
  -H "Content-Type: application/json" \
  --data @"$SCRIPT_DIR/sample_customer.json")"
echo "Prediction: $response"
echo "$response" | grep -q '"churn_probability"'

# Invalid input must be rejected with 422, not scored.
code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://localhost:$PORT/predict" \
  -H "Content-Type: application/json" --data '{"tenure": -5}')"
[ "$code" = "422" ] || { echo "Expected 422 for invalid input, got $code"; exit 1; }

echo "Smoke test passed."
